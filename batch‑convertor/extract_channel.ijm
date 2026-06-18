// Extract Single Channel from Multi-Channel TIF Stack
// Processes TIF files whose crop number falls within [crop_start, crop_end] (both included).
// Crop number is read from the filename, e.g. "crop17_..." -> 17.
// Output saved to "extracted_chN" subfolder inside the source folder.

requires("1.53c");
run("Close All");

#@ File    (label="Select folder containing TIF files", style="directory") input_dir
#@ Integer (label="Channel to extract (1-indexed):", value=2, min=1, max=10) channel_num
#@ Integer (label="First crop number to include:", value=1,  min=1) crop_start
#@ Integer (label="Last crop number to include:",  value=24, min=1) crop_end

t_start = getTime();
setBatchMode(true);
print("\\Clear");
print("Channel: " + channel_num + "   Crop range: " + crop_start + " - " + crop_end + " (inclusive)");

sep = File.separator;
output_dir = input_dir + sep + "extracted_ch" + channel_num;
File.makeDirectory(output_dir);

filelist = getFileList(input_dir);
processed    = 0;
skipped      = 0;
out_of_range = 0;

for (i = 0; i < filelist.length; i++) {
    fname = filelist[i];
    if (!endsWith(fname, ".tif") && !endsWith(fname, ".tiff")) continue;

    // extract the first run of digits after "crop" in the filename
    lower   = toLowerCase(fname);
    idx     = indexOf(lower, "crop");
    if (idx < 0) {
        print("SKIPPED (no 'crop' in name): " + fname);
        skipped++;
        continue;
    }
    num_str = "";
    pos = idx + 4;
    while (pos < lengthOf(fname) && charCodeAt(fname, pos) >= 48 && charCodeAt(fname, pos) <= 57) {
        num_str = num_str + substring(fname, pos, pos + 1);
        pos++;
    }
    if (lengthOf(num_str) == 0) {
        print("SKIPPED (no number after 'crop'): " + fname);
        skipped++;
        continue;
    }
    crop_num = parseInt(num_str);

    if (crop_num < crop_start || crop_num > crop_end) {
        out_of_range++;
        continue;
    }

    open(input_dir + sep + fname);
    orig_title = getTitle();
    Stack.getDimensions(width, height, channels, slices, frames);

    if (channel_num <= channels) {
        run("Make Substack...", "channels=" + channel_num);
        saveAs("Tiff", output_dir + sep + fname);
        close();
        selectWindow(orig_title);
        close();
        processed++;
        print("\\Update1: [" + processed + "] crop" + crop_num + "  " + fname);
    } else {
        close();
        skipped++;
        print("SKIPPED (only " + channels + " ch, requested " + channel_num + "): " + fname);
    }
}

setBatchMode(false);
elapsed = (getTime() - t_start) / 1000;
print("Done.  Processed: " + processed + "   Out of range: " + out_of_range + "   Skipped: " + skipped);
print("Output: " + output_dir);
print("Time: " + elapsed + " s");
