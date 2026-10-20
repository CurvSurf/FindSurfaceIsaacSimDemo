"""UI-thread preview/selection state and immutable capture collection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from threading import Lock, get_ident
from time import monotonic
from typing import Callable, Protocol

from .camera import CameraSnapshot
from .detection import (
    DetectionCompletion,
    DetectionCompletionStatus,
    DetectionSubmit,
    DetectionTelemetry,
    SerializedDetectionScheduler,
)
from .domain import (
    CapturedSurface,
    DetectionOptions,
    DetectionRequest,
    DetectionRequestKey,
    DetectionResult,
    GenerationSnapshot,
    ScanFrame,
    SelectionSnapshot,
)
from .picking import PickResult, PickingController, PickingOverlay, PickingTransition
from .render_layers import build_seed_layer


class DetectionSchedulerPort(Protocol):
    def update_current(self, generations: GenerationSnapshot, frame) -> None: ...
    def submit(self, request: DetectionRequest) -> DetectionSubmit: ...
    def acknowledge_completion(
        self, request_key: DetectionRequestKey, generations: GenerationSnapshot,
    ) -> bool: ...
    def discard_pending_retry(self) -> None: ...
    def close(self, timeout_s: float | None = None) -> None: ...
    def telemetry(self) -> DetectionTelemetry: ...


class PreviewPhase(str, Enum):
    NO_SEED = "NO_SEED"
    PENDING = "PENDING"
    READY = "READY"
    ERROR = "ERROR"
    CLOSED = "CLOSED"


class CaptureBlockReason(str, Enum):
    SENSOR_ACTIVE = "sensor_active"
    SELECTION_REQUIRED = "selection_required"
    DETECTION_PENDING = "detection_pending"
    DETECTION_FAILED = "detection_failed"
    RESULT_STALE = "result_stale"
    NO_SUCCESSFUL_RESULT = "no_successful_result"
    CLOSED = "closed"


_CAPTURE_REASON_TEXT = {
    CaptureBlockReason.SENSOR_ACTIVE: "Disable the sensor before capturing.",
    CaptureBlockReason.SELECTION_REQUIRED: "Select a point in the fixed cloud before capturing.",
    CaptureBlockReason.DETECTION_PENDING: "Wait for the selected-point detection to finish.",
    CaptureBlockReason.DETECTION_FAILED: "The selected-point detection failed; adjust options and retry.",
    CaptureBlockReason.RESULT_STALE: "The detection result no longer matches the current frame, selection, or options.",
    CaptureBlockReason.NO_SUCCESSFUL_RESULT: "No successful selected-point detection is available.",
    CaptureBlockReason.CLOSED: "The preview and capture workflow is closed.",
}


@dataclass(frozen=True)
class CaptureAvailability:
    enabled: bool
    reasons: tuple[CaptureBlockReason, ...]
    tooltip: str


@dataclass(frozen=True)
class CaptureEntry:
    surface: CapturedSurface
    visible: bool = True

    @property
    def capture_id(self) -> str:
        return self.surface.capture_id


@dataclass(frozen=True)
class PreviewCaptureSnapshot:
    phase: PreviewPhase
    sensor_active: bool
    point_selection_mode: bool
    request_key: DetectionRequestKey | None
    result: DetectionResult | None
    display_result: DetectionResult | None
    display_result_sticky: bool
    display_seed_used_fallback: bool
    error: str | None
    captures: tuple[CaptureEntry, ...]
    selected_capture_id: str | None
    capture_availability: CaptureAvailability
    completion_mailbox_overwrites: int


class PreviewCaptureController:
    """Combine picking, scheduler completion marshalling, and stable captures."""

    def __init__(
        self,
        picking: PickingController,
        scheduler: DetectionSchedulerPort,
        frame: ScanFrame,
        options: DetectionOptions,
        *,
        owns_scheduler: bool = False,
        sticky_timeout_s: float | None = None,
    ) -> None:
        self.picking = picking
        self._scheduler = scheduler
        self._owns_scheduler = owns_scheduler
        self._frame = frame
        self._options = options
        self._phase = PreviewPhase.NO_SEED
        self._current_request: DetectionRequest | None = None
        self._result: DetectionResult | None = None
        self._display_result: DetectionResult | None = None
        self._display_request: DetectionRequest | None = None
        # Demo 1's retroactive pick survives visible Sticky preview expiration.
        # Keep references, not copies of the immutable cloud/inlier buffers.
        self._last_successful_result: DetectionResult | None = None
        self._last_successful_request: DetectionRequest | None = None
        self._completion_selection_floor = 0
        self._sticky_timeout_s = None if sticky_timeout_s is None else max(0.1, float(sticky_timeout_s))
        self._sticky_deadline: float | None = None
        self._error: str | None = None
        self._captures: list[CaptureEntry] = []
        self._selected_capture_id: str | None = None
        self._capture_serial = 0
        self._completion_lock = Lock()
        self._completion_mailbox: DetectionCompletion | None = None
        self._completion_mailbox_overwrites = 0
        self._pending_completion_ack: tuple[DetectionRequestKey, GenerationSnapshot] | None = None
        self._owner_thread = get_ident()
        self._closed = False
        self._scheduler.update_current(picking.state.generations, frame.identity)

    @classmethod
    def create(
        cls,
        picking: PickingController,
        detector_factory: Callable,
        frame: ScanFrame,
        options: DetectionOptions,
        *,
        sticky_timeout_s: float | None = None,
    ) -> "PreviewCaptureController":
        target: list[PreviewCaptureController] = []

        def on_completion(completion: DetectionCompletion) -> None:
            target[0].enqueue_completion(completion)

        scheduler = SerializedDetectionScheduler(
            detector_factory,
            current_generations=picking.state.generations,
            current_frame=frame.identity,
            on_completion=on_completion,
            completion_ack_required=True,
        )
        controller = cls(
            picking, scheduler, frame, options,
            owns_scheduler=True,
            sticky_timeout_s=sticky_timeout_s,
        )
        target.append(controller)
        controller.sync_current()
        return controller

    @property
    def frame(self) -> ScanFrame:
        return self._frame

    def detection_telemetry(self) -> DetectionTelemetry:
        return self._scheduler.telemetry()

    def successful_call_mean_ms(self, window: int) -> float | None:
        return self._scheduler.successful_call_mean_ms(window)

    @property
    def options(self) -> DetectionOptions:
        return self._options

    @property
    def retroactive_seed(self) -> SelectionSnapshot | None:
        request = self._last_successful_request
        if request is None or not self._same_source_context(request):
            return None
        return request.selection

    def pick_overlay(self, *, retroactive: bool = False, show_hover_radius: bool = True) -> PickingOverlay:
        """Alt changes the displayed/clickable seed, never the detection input."""

        overlay = self.picking.overlay(
            seed_radius_m=self._options.seed_radius_m, show_hover_radius=show_hover_radius,
        )
        seed = self.retroactive_seed if retroactive else None
        if seed is None or self.picking.state.selection is not None:
            return overlay
        return replace(overlay, seed=build_seed_layer(
            seed.seed_world_m, used_fallback=seed.used_fallback,
        ))

    @property
    def snapshot(self) -> PreviewCaptureSnapshot:
        with self._completion_lock:
            overwrites = self._completion_mailbox_overwrites
        return PreviewCaptureSnapshot(
            phase=self._phase,
            sensor_active=self.picking.state.sensor_active,
            point_selection_mode=self.picking.state.point_selection_mode,
            request_key=None if self._current_request is None else self._current_request.key,
            result=self._result,
            display_result=self._display_result,
            display_result_sticky=(
                self._display_result is not None and self._display_result is not self._result
            ),
            display_seed_used_fallback=(
                False if self._display_request is None
                else self._display_request.selection.used_fallback
            ),
            error=self._error,
            captures=tuple(self._captures),
            selected_capture_id=self._selected_capture_id,
            capture_availability=self.capture_availability(),
            completion_mailbox_overwrites=overwrites,
        )

    def sync_current(self) -> DetectionSubmit | None:
        self._require_owner()
        self._require_open()
        return self._sync_preview()

    def update_hover(self, cursor_logical: tuple[float, float]) -> PickingTransition:
        self._require_owner()
        transition = self.picking.update_hover(cursor_logical)
        self._sync_preview()
        return transition

    def clear_hover(self) -> PickingTransition:
        self._require_owner()
        transition = self.picking.clear_hover()
        self._sync_preview()
        return transition

    def select_click(self, cursor_logical: tuple[float, float] | None = None) -> PickingTransition:
        self._require_owner()
        reusable_result = self._result
        reusable_request = self._current_request
        transition = self.picking.select_click(cursor_logical)
        selection = self.picking.state.selection
        if (
            selection is not None
            and reusable_result is not None
            and reusable_request is not None
            and reusable_request.frame is self._frame
            and reusable_request.selection.seed_index == selection.seed_index
            and reusable_request.options == self._options
        ):
            request = DetectionRequest(
                self._frame, selection, self._options, self.picking.state.generations,
            )
            self._scheduler.update_current(request.generations, self._frame.identity)
            self._current_request = request
            self._result = replace(
                reusable_result,
                request_key=request.key,
                generations=request.generations,
            )
            self._display_result = self._result
            self._display_request = request
            self._sticky_deadline = None
            self._phase = PreviewPhase.READY
            self._error = None
            return transition
        self._sync_preview()
        return transition

    def select_sticky(self) -> PickingTransition:
        """Compatibility entry point for the original Sticky selection adapter."""

        return self.select_last_successful()

    def select_last_successful(self) -> PickingTransition:
        """Alt+click uses the last success, even after visible Sticky expiry."""

        self._require_owner()
        result, request = self._last_successful_result, self._last_successful_request
        if (
            self.picking.state.sensor_active or self.retroactive_seed is None
            or result is None or request is None
        ):
            return PickingTransition(self.picking.state)
        transition = self.picking.select_index(
            result.request_key.seed_index,
            used_fallback=request.selection.used_fallback,
        )
        selection = self.picking.state.selection
        if selection is None:
            return transition
        if request.options != self._options:
            self._sync_preview()
            return transition
        promoted = DetectionRequest(
            self._frame, selection, self._options, self.picking.state.generations,
        )
        self._scheduler.update_current(promoted.generations, self._frame.identity)
        self._scheduler.discard_pending_retry()
        self._current_request = promoted
        self._result = replace(
            result,
            request_key=promoted.key,
            generations=promoted.generations,
        )
        self._display_result = self._result
        self._display_request = promoted
        self._sticky_deadline = None
        self._phase = PreviewPhase.READY
        self._error = None
        return transition

    def cancel_selection(self) -> PickingTransition:
        self._require_owner()
        transition = self.picking.cancel_selection()
        self._sync_preview()
        return transition

    def pause_detection(self) -> PickingTransition:
        """Clear transient detection state without blocking on the worker.

        Demo 1's pointer/inspect tool pauses preview detection immediately.  A
        native call that is already running is allowed to finish, but its
        completion cannot restore the cleared preview and no latest retry is
        retained.
        """

        self._require_owner()
        self._require_open()
        transition = self.picking.clear_interaction()
        current = self.picking.state.generations
        self._completion_selection_floor = current.selection_revision + 1
        self._scheduler.update_current(current, self._frame.identity)
        self._scheduler.discard_pending_retry()
        self._phase = PreviewPhase.NO_SEED
        self._current_request = None
        self._result = None
        self._clear_display_result()
        self._clear_last_successful()
        self._error = None
        return transition

    def update_context(
        self,
        frame: ScanFrame,
        camera: CameraSnapshot,
        logical_size: tuple[float, float],
        generations: GenerationSnapshot,
        options: DetectionOptions,
        *,
        sensor_active: bool,
    ) -> PickingTransition:
        self._require_owner()
        if generations.options_revision != options.revision:
            raise ValueError("generation options_revision must match detection options")
        transition = self.picking.update_context(
            frame,
            camera,
            logical_size,
            generations,
            sensor_active=sensor_active,
        )
        self._frame = frame
        self._options = options
        if self._last_successful_request is not None and not self._same_source_context(self._last_successful_request):
            self._clear_last_successful()
        if self._display_request is not None and not self._same_durable_context(self._display_request):
            self._clear_display_result()
        elif self._display_request is not None and not self._same_source_context(self._display_request):
            # A live scan may supersede the immutable frame used by a successful
            # native call. Keep its geometry as display-only Sticky state while
            # the scheduler collapses later frames to one latest retry.
            self._arm_sticky_deadline()
        self._sync_preview()
        return transition

    def enqueue_completion(self, completion: DetectionCompletion) -> None:
        """Worker-thread callback: replace one mailbox item without touching UI state."""

        with self._completion_lock:
            if self._closed:
                return
            if self._completion_mailbox is not None:
                self._completion_mailbox_overwrites += 1
            self._completion_mailbox = completion

    def drain_completion(self) -> bool:
        """Apply the latest scheduler completion on the controller/UI owner thread."""

        self._require_owner()
        self._require_open()
        with self._completion_lock:
            completion, self._completion_mailbox = self._completion_mailbox, None
        if completion is None:
            return False
        self._pending_completion_ack = completion.request_key, completion.generations
        request = self._current_request
        if (
            request is None
            or completion.request_key != request.key
            or completion.generations != request.generations
        ):
            return self._present_superseded_hover_completion(completion)
        if completion.status is DetectionCompletionStatus.PUBLISHED and completion.result is not None:
            self._phase = PreviewPhase.READY
            self._result = completion.result
            self._display_result = completion.result
            self._display_request = request
            self._remember_success(completion.result, request)
            self._sticky_deadline = None
            self._error = None
        elif completion.status is DetectionCompletionStatus.FAILED:
            self._phase = PreviewPhase.ERROR
            self._result = None
            self._error = completion.error or "FindSurface detection failed"
            if self.picking.state.selection is not None:
                self._clear_display_result()
            else:
                # A stream of failures must not extend an old successful result
                # indefinitely. The deadline starts once, when that result first
                # becomes stale, and subsequent failures preserve it unchanged.
                self._arm_sticky_deadline()
        else:
            self._phase = PreviewPhase.PENDING
            self._result = None
            self._error = None
        return True

    def acknowledge_completion(self) -> bool:
        """Allow the worker to run the one latest retry after UI presentation."""

        self._require_owner()
        token, self._pending_completion_ack = self._pending_completion_ack, None
        if token is None:
            return False
        return self._scheduler.acknowledge_completion(*token)

    def _present_superseded_hover_completion(self, completion: DetectionCompletion) -> bool:
        """Show a completed hover while the latest-only retry follows the cursor.

        Demo 1 does not replace the active request key while FindSurface is busy:
        it presents that completion, then starts the one deferred latest seed.  The
        scheduler here updates its generation gate eagerly, so the equivalent
        completion arrives as selection/options-stale.  It is safe as display-only
        Sticky state while the immutable frame and durable generations are still
        current; it must never become the authoritative result used by Capture.
        """

        completed_request = completion.request
        result = completion.result
        if (
            completed_request is None
            or result is None
            or completion.request_key != completed_request.key
            or completion.generations != completed_request.generations
            or result.request_key != completed_request.key
            or result.generations != completed_request.generations
            or completed_request.generations.selection_revision < self._completion_selection_floor
        ):
            return False
        current = self.picking.state.generations
        observed = completed_request.generations
        if any(
            getattr(observed, axis) != getattr(current, axis)
            for axis in ("session", "scene", "sensor")
        ):
            return False
        # Hover selection, FindSurface-option and live-frame supersession are
        # display-only. The immutable result geometry is still valid to show,
        # but remains ineligible for Capture and old-frame point classification.
        # The current request stays PENDING, and UI acknowledgement releases the
        # capacity-one latest request only after this result has been presented.
        self._display_result = result
        self._display_request = completed_request
        self._remember_success(result, completed_request)
        if self._sticky_timeout_s is None:
            # Even when timed Sticky retention is disabled, the just-completed
            # result must survive this UI pass.  The acknowledged latest retry
            # will replace it with the current request's completion.
            self._sticky_deadline = None
        else:
            self._arm_sticky_deadline(reset=True)
        return True

    def expire_sticky(self, now_s: float | None = None) -> bool:
        """Expire a first-demo-style last-successful hover preview."""

        self._require_owner()
        deadline = self._sticky_deadline
        if deadline is None or self._display_result is None:
            return False
        if (monotonic() if now_s is None else float(now_s)) < deadline:
            return False
        self._clear_display_result()
        return True

    def configure_sticky(self, *, enabled: bool, timeout_s: float) -> bool:
        """Apply the demo-1 Sticky policy and report a visible-result change."""

        self._require_owner()
        timeout = max(0.1, float(timeout_s)) if enabled else None
        if timeout == self._sticky_timeout_s:
            return False
        was_visible = self._display_result is not None
        self._sticky_timeout_s = timeout
        if timeout is None:
            if self.snapshot.display_result_sticky:
                self._clear_display_result()
        elif self.snapshot.display_result_sticky:
            self._arm_sticky_deadline(reset=True)
        return was_visible != (self._display_result is not None)

    def capture_availability(self) -> CaptureAvailability:
        reasons: list[CaptureBlockReason] = []
        if self._closed:
            reasons.append(CaptureBlockReason.CLOSED)
        if self.picking.state.sensor_active:
            reasons.append(CaptureBlockReason.SENSOR_ACTIVE)
        selection = self.picking.state.selection
        if not self.picking.state.point_selection_mode or selection is None:
            reasons.append(CaptureBlockReason.SELECTION_REQUIRED)
        request = self._current_request
        if self._phase is PreviewPhase.PENDING:
            reasons.append(CaptureBlockReason.DETECTION_PENDING)
        elif self._phase is PreviewPhase.ERROR:
            reasons.append(CaptureBlockReason.DETECTION_FAILED)
        elif self._result is None:
            reasons.append(CaptureBlockReason.NO_SUCCESSFUL_RESULT)
        if self._result is not None and not self._result_matches_current_selection(request, selection):
            reasons.append(CaptureBlockReason.RESULT_STALE)
        unique = tuple(dict.fromkeys(reasons))
        return CaptureAvailability(
            enabled=not unique,
            reasons=unique,
            tooltip="Capture the selected detection." if not unique else " ".join(_CAPTURE_REASON_TEXT[item] for item in unique),
        )

    def capture(self) -> CaptureEntry:
        self._require_owner()
        availability = self.capture_availability()
        if not availability.enabled:
            raise RuntimeError(availability.tooltip)
        request, result = self._current_request, self._result
        if request is None or result is None:
            raise AssertionError("capture availability accepted without a current result")
        self._capture_serial += 1
        capture_id = f"capture-{self._capture_serial:06d}"
        surface = CapturedSurface.from_detection(
            capture_id,
            result,
            self._frame,
            request=request,
        )
        entry = CaptureEntry(surface)
        replacement_index = next(
            (
                index for index, existing in enumerate(self._captures)
                if existing.surface.geometry_type is surface.geometry_type
            ),
            None,
        )
        if replacement_index is None:
            self._captures.append(entry)
        else:
            self._captures[replacement_index] = entry
        self._selected_capture_id = capture_id
        self.picking.clear_interaction()
        self._completion_selection_floor = self.picking.state.generations.selection_revision
        self._clear_last_successful()
        self._sync_preview()
        return entry

    def set_capture_visibility(self, capture_id: str, visible: bool) -> CaptureEntry:
        self._require_owner()
        index = self._capture_index(capture_id)
        self._captures[index] = replace(self._captures[index], visible=bool(visible))
        return self._captures[index]

    def select_capture(self, capture_id: str | None) -> None:
        self._require_owner()
        if capture_id is not None:
            self._capture_index(capture_id)
        self._selected_capture_id = capture_id

    def delete_capture(self, capture_id: str) -> CaptureEntry:
        self._require_owner()
        index = self._capture_index(capture_id)
        entry = self._captures.pop(index)
        if self._selected_capture_id == capture_id:
            self._selected_capture_id = None
        return entry

    def clear_captures(self) -> tuple[CaptureEntry, ...]:
        """Clear controller-side capture bookkeeping without rewinding its ID serial."""

        self._require_owner()
        removed = tuple(self._captures)
        self._captures.clear()
        self._selected_capture_id = None
        return removed

    def close(self, timeout_s: float | None = None) -> None:
        self._require_owner()
        if self._closed:
            return
        self._closed = True
        with self._completion_lock:
            self._completion_mailbox = None
        self._pending_completion_ack = None
        if self._owns_scheduler:
            self._scheduler.close(timeout_s)
        self._phase = PreviewPhase.CLOSED
        self._current_request = None
        self._result = None
        self._clear_display_result()
        self._clear_last_successful()
        self._error = None

    def _same_source_context(self, request: DetectionRequest) -> bool:
        current = self.picking.state.generations
        return request.frame is self._frame and all(
            getattr(request.generations, axis) == getattr(current, axis)
            for axis in ("session", "scene", "sensor", "frame")
        )

    def _same_durable_context(self, request: DetectionRequest) -> bool:
        current = self.picking.state.generations
        return all(
            getattr(request.generations, axis) == getattr(current, axis)
            for axis in ("session", "scene", "sensor")
        )

    def _remember_success(self, result: DetectionResult, request: DetectionRequest) -> None:
        if result.inlier_indices and self._same_source_context(request):
            self._last_successful_result = result
            self._last_successful_request = request

    def _clear_last_successful(self) -> None:
        self._last_successful_result = None
        self._last_successful_request = None

    def _sync_preview(self) -> DetectionSubmit | None:
        state = self.picking.state
        if state.frame_identity != self._frame.identity:
            raise ValueError("picking state must reference the current preview frame")
        if state.generations.options_revision != self._options.revision:
            raise ValueError("picking generation must match detection option revision")
        self._scheduler.update_current(state.generations, self._frame.identity)
        selected = state.selection
        displayed = selected if selected is not None else state.hover
        if displayed is None:
            self._scheduler.discard_pending_retry()
            self._phase = PreviewPhase.NO_SEED
            self._current_request = None
            self._result = None
            self._error = None
            self._arm_sticky_deadline()
            return None
        selection = selected or self._hover_selection(displayed, state.generations, self._frame)
        request = DetectionRequest(self._frame, selection, self._options, state.generations)
        if (
            self._current_request is not None
            and self._current_request.key == request.key
            and self._current_request.generations == request.generations
            and self._phase in {PreviewPhase.PENDING, PreviewPhase.READY, PreviewPhase.ERROR}
        ):
            return None
        self._current_request = request
        self._phase = PreviewPhase.PENDING
        self._result = None
        self._error = None
        if self.picking.state.selection is not None:
            self._clear_display_result()
        else:
            self._arm_sticky_deadline()
        return self._scheduler.submit(request)

    def _arm_sticky_deadline(self, *, reset: bool = False) -> None:
        if self._display_result is None:
            self._sticky_deadline = None
        elif self._sticky_timeout_s is None:
            self._clear_display_result()
        elif reset or self._sticky_deadline is None:
            self._sticky_deadline = monotonic() + self._sticky_timeout_s

    def _clear_display_result(self) -> None:
        self._display_result = None
        self._display_request = None
        self._sticky_deadline = None

    @staticmethod
    def _hover_selection(
        result: PickResult,
        generations: GenerationSnapshot,
        frame: ScanFrame,
    ) -> SelectionSnapshot:
        return SelectionSnapshot(
            generations.selection_revision,
            frame.identity,
            seed_index=result.point_index,
            seed_world_m=result.point_world_m,
            used_fallback=result.used_fallback,
        )

    def _result_matches_current_selection(
        self,
        request: DetectionRequest | None,
        selection: SelectionSnapshot | None,
    ) -> bool:
        return bool(
            request is not None
            and selection is not None
            and self._result is not None
            and request.frame is self._frame
            and request.selection == selection
            and request.options == self._options
            and request.generations == self.picking.state.generations
            and self._result.request_key == request.key
            and self._result.generations == request.generations
        )

    def _capture_index(self, capture_id: str) -> int:
        for index, entry in enumerate(self._captures):
            if entry.capture_id == capture_id:
                return index
        raise KeyError(f"unknown capture id: {capture_id}")

    def _require_owner(self) -> None:
        if get_ident() != self._owner_thread:
            raise RuntimeError("preview/capture state must be mutated on its owner UI thread")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("preview/capture controller is closed")


__all__ = [
    "CaptureAvailability",
    "CaptureBlockReason",
    "CaptureEntry",
    "DetectionSchedulerPort",
    "PreviewCaptureController",
    "PreviewCaptureSnapshot",
    "PreviewPhase",
]
