local ext = get_current_extension_info()
project_ext(ext)
repo_build.prebuild_link {
    { "config", ext.target_dir.."/config" },
    { "docs", ext.target_dir.."/docs" },
    { "data", ext.target_dir.."/data" },
    { "findsurface_isaac_sim_launcher", ext.target_dir.."/findsurface_isaac_sim_launcher" },
}
