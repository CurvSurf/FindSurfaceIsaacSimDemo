using System;
using System.IO;
using System.IO.Compression;
using System.Collections.Generic;
using System.Diagnostics;

namespace CurvSurf.Installation
{
    // Extract only into a new staging directory. The caller verifies the ZIP hash.
    public static class RuntimeZipExtractor
    {
        private static string LongPath(string path)
        {
            return path.StartsWith(@"\\")
                ? @"\\?\UNC\" + path.Substring(2)
                : @"\\?\" + path;
        }

        private static string RelativePath(ZipArchiveEntry entry)
        {
            string name = entry.FullName.Replace('/', '\\');
            if (String.IsNullOrEmpty(name) || name.StartsWith(@"\") || name.Contains(":"))
                throw new InvalidDataException("Unsafe ZIP entry: " + entry.FullName);
            // Unix symlinks must never become links into another installation.
            if (((entry.ExternalAttributes >> 16) & 0xF000) == 0xA000)
                throw new InvalidDataException("ZIP symlink is not supported: " + entry.FullName);
            string relative = name.TrimEnd('\\');
            foreach (string part in relative.Split('\\'))
            {
                if (part.Length == 0 || part == "." || part == ".." ||
                    part.EndsWith(".") || part.EndsWith(" ") ||
                    part.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
                    throw new InvalidDataException("Unsafe ZIP entry: " + entry.FullName);
                string stem = part.Split('.')[0].ToUpperInvariant();
                if (stem == "CON" || stem == "PRN" || stem == "AUX" || stem == "NUL" ||
                    (stem.Length == 4 && (stem.StartsWith("COM") || stem.StartsWith("LPT")) &&
                     stem[3] >= '1' && stem[3] <= '9'))
                    throw new InvalidDataException("Reserved ZIP filename: " + entry.FullName);
            }
            return relative;
        }

        public static bool ContainsReparsePoints(string directory)
        {
            var pending = new Stack<string>();
            pending.Push(LongPath(Path.GetFullPath(directory).TrimEnd('\\')));
            while (pending.Count > 0)
            {
                string current = pending.Pop();
                if ((File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
                    return true;
                foreach (string child in Directory.EnumerateFileSystemEntries(current))
                {
                    FileAttributes attributes = File.GetAttributes(child);
                    if ((attributes & FileAttributes.ReparsePoint) != 0)
                        return true;
                    if ((attributes & FileAttributes.Directory) != 0)
                        pending.Push(child);
                }
            }
            return false;
        }

        // Only called after the ownership and no-link guards have passed.
        public static void DeleteOwnedDirectory(string directory)
        {
            Directory.Delete(LongPath(Path.GetFullPath(directory).TrimEnd('\\')), true);
        }

        public static int Extract(string archivePath, string destination)
        {
            return Extract(archivePath, destination, null);
        }

        public static int Extract(string archivePath, string destination,
            Action<string, long, long, int, int> progress)
        {
            string root = Path.GetFullPath(destination).TrimEnd('\\');
            string extendedRoot = LongPath(root);
            if (Directory.Exists(extendedRoot) || File.Exists(extendedRoot))
                throw new IOException("Extraction destination must not exist: " + root);
            using (ZipArchive archive = ZipFile.OpenRead(archivePath))
            {
                var clock = Stopwatch.StartNew();
                long lastReport = 0;
                long totalBytes = 0;
                int totalFiles = 0;
                int checkedEntries = 0;
                if (progress != null)
                    progress("Checking ZIP paths", 0, 0, 0, archive.Entries.Count);
                // Validate all names before writing any payload.
                foreach (ZipArchiveEntry entry in archive.Entries)
                {
                    RelativePath(entry);
                    if (!entry.FullName.EndsWith("/") && !entry.FullName.EndsWith(@"\"))
                    {
                        totalBytes = checked(totalBytes + entry.Length);
                        totalFiles++;
                    }
                    checkedEntries++;
                    if (progress != null && clock.ElapsedMilliseconds - lastReport >= 1000)
                    {
                        progress("Checking ZIP paths", 0, 0, checkedEntries, archive.Entries.Count);
                        lastReport = clock.ElapsedMilliseconds;
                    }
                }
                Directory.CreateDirectory(extendedRoot);
                int files = 0;
                long completedBytes = 0;
                byte[] buffer = new byte[131072];
                lastReport = clock.ElapsedMilliseconds;
                if (progress != null)
                    progress("Extracting ZIP", 0, totalBytes, 0, totalFiles);
                foreach (ZipArchiveEntry entry in archive.Entries)
                {
                    string target = extendedRoot + @"\" + RelativePath(entry);
                    if (entry.FullName.EndsWith("/") || entry.FullName.EndsWith(@"\"))
                    {
                        Directory.CreateDirectory(target);
                        continue;
                    }
                    Directory.CreateDirectory(Path.GetDirectoryName(target));
                    using (Stream input = entry.Open())
                    using (FileStream output = new FileStream(target, FileMode.CreateNew,
                        FileAccess.Write, FileShare.None, 131072))
                    {
                        int count;
                        while ((count = input.Read(buffer, 0, buffer.Length)) != 0)
                        {
                            output.Write(buffer, 0, count);
                            completedBytes += count;
                            if (progress != null && clock.ElapsedMilliseconds - lastReport >= 1000)
                            {
                                progress("Extracting ZIP", completedBytes, totalBytes, files, totalFiles);
                                lastReport = clock.ElapsedMilliseconds;
                            }
                        }
                        if (output.Length != entry.Length)
                            throw new InvalidDataException("Incomplete ZIP entry: " + entry.FullName);
                    }
                    files++;
                    // Empty/small files still provide a heartbeat on slow disks.
                    if (progress != null && clock.ElapsedMilliseconds - lastReport >= 1000)
                    {
                        progress("Extracting ZIP", completedBytes, totalBytes, files, totalFiles);
                        lastReport = clock.ElapsedMilliseconds;
                    }
                }
                if (progress != null)
                    progress("Extracting ZIP", completedBytes, totalBytes, files, totalFiles);
                return files;
            }
        }
    }
}
