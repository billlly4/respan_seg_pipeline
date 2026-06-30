// Extract a single channel from every TIF in a folder.
// Channel index is 0-based (0 = first channel, 1 = second, etc.).
// Output saved to "extracted_ch{N}" subfolder with the same filenames.

requires("1.53c");
run("Close All");

#@ File    (label="Select folder containing TIF files", style="directory") input_dir
#@ Integer (label="Channel to extract (0 = first, 1 = second, ...):", value=0, min=0, max=9) channel_idx

fiji_ch = channel_idx + 1;   // Fiji channels are 1-indexed internally

t_start = getTime();
setBatchMode(true);
print("\\Clear");
print("Extracting channel " + channel_idx + " from all TIFs in: " + input_dir);

sep = File.separator;
output_dir = input_dir + sep + "extracted_ch" + channel_idx;
File.makeDirectory(output_dir);

filelist = getFileList(input_dir);
processed = 0;
skipped   = 0;

for (i = 0; i < filelist.length; i++) {
    fname = filelist[i];
    if (!endsWith(fname, ".tif") && !endsWith(fname, ".tiff")) continue;

    open(input_dir + sep + fname);
    orig_title = getTitle();
    Stack.getDimensions(width, height, channels, slices, frames);

    if (fiji_ch <= channels) {
        run("Make Substack...", "channels=" + fiji_ch);
        saveAs("Tiff", output_dir + sep + fname);
        close();
        selectWindow(orig_title);
        close();
        processed++;
        print("[" + processed + "] " + fname);
    } else {
        close();
        skipped++;
        print("SKIPPED (only " + channels + " channel(s), requested index " + channel_idx + "): " + fname);
    }
}

setBatchMode(false);
elapsed = (getTime() - t_start) / 1000;
print("Done.  Processed: " + processed + "   Skipped: " + skipped);
print("Output: " + output_dir);
print("Time: " + elapsed + " s");
