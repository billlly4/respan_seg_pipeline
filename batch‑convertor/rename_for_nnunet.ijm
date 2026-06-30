// Rename TIF files in imagesTr/ and labelsTr/ to nnU-Net case naming format.
//
//   Images -> case{NNN}_0000.tif
//   Labels -> case{NNN}.tif
//
// Files are paired by alphabetical sort order — both folders must contain the
// same number of TIF files and be in the same order.
// Files already named case{NNN}... are skipped automatically.
// New case numbers continue from the highest existing case number found.

requires("1.53c");

#@ File (label="Images folder (imagesTr)", style="directory") images_dir
#@ File (label="Labels folder (labelsTr)", style="directory") labels_dir

sep = File.separator;
print("\\Clear");

img_files = getNewTifList(images_dir);
lbl_files = getNewTifList(labels_dir);

n = img_files.length;

if (n == 0) {
    print("No new (non-case) TIF files found — nothing to rename.");
    exit();
}

if (lbl_files.length != n) {
    exit("Mismatch: " + n + " new image(s) but " + lbl_files.length +
         " new label(s). Both folders must have the same number of new TIF files.");
}

next_case = findNextCaseNumber(images_dir);
print("New files found: " + n);
print("Starting from case number: " + next_case);
print("");

for (i = 0; i < n; i++) {
    case_str = zeroPad(next_case + i, 3);

    old_img = images_dir + sep + img_files[i];
    new_img = images_dir + sep + "case" + case_str + "_0000.tif";

    old_lbl = labels_dir + sep + lbl_files[i];
    new_lbl = labels_dir + sep + "case" + case_str + ".tif";

    ok_img = File.rename(old_img, new_img);
    ok_lbl = File.rename(old_lbl, new_lbl);

    if (ok_img && ok_lbl) {
        print("[" + (i + 1) + "]  " + img_files[i] + "  ->  case" + case_str);
    } else {
        print("ERROR on case" + case_str +
              "  (image=" + ok_img + "  label=" + ok_lbl + ")");
    }
}

print("");
print("Done.  Renamed " + n + " pair(s).");


// ---------------------------------------------------------------------------
// Helper: return sorted list of TIFs that are NOT already named case{NNN}...
// ---------------------------------------------------------------------------
function getNewTifList(dir) {
    all = getFileList(dir);
    setOption("ExpandableArrays", true);
    result = newArray;
    k = 0;
    for (i = 0; i < all.length; i++) {
        fname = all[i];
        lower = toLowerCase(fname);
        is_tif  = endsWith(lower, ".tif") || endsWith(lower, ".tiff");
        is_case = (indexOf(lower, "case") == 0);
        if (is_tif && !is_case) {
            result[k] = fname;
            k++;
        }
    }
    result = Array.sort(result);
    return result;
}

// ---------------------------------------------------------------------------
// Helper: find next available case number by scanning imagesTr for case*_0000
// ---------------------------------------------------------------------------
function findNextCaseNumber(dir) {
    all = getFileList(dir);
    max_num = -1;
    for (i = 0; i < all.length; i++) {
        lower = toLowerCase(all[i]);
        idx_suffix = indexOf(lower, "_0000");
        if (indexOf(lower, "case") == 0 && idx_suffix > 4) {
            num_str = substring(all[i], 4, idx_suffix);
            num = parseInt(num_str);
            if (!isNaN(num) && num > max_num) {
                max_num = num;
            }
        }
    }
    return max_num + 1;   // 0 if no existing cases, otherwise max+1
}

// ---------------------------------------------------------------------------
// Helper: zero-pad an integer to `digits` characters
// ---------------------------------------------------------------------------
function zeroPad(num, digits) {
    s = "" + num;
    while (lengthOf(s) < digits) {
        s = "0" + s;
    }
    return s;
}
