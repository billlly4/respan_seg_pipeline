// Add the nnU-Net channel suffix _0000 to every image in imagesTr/

requires("1.53c");

#@ File (label="Images folder (imagesTr)", style="directory") images_dir

sep = File.separator;
print("\\Clear");
print("Adding _0000 to images in: " + images_dir);
print("");

all = getFileList(images_dir);
n = 0;
skipped = 0;

for (i = 0; i < all.length; i++) {
    fname = all[i];
    lower = toLowerCase(fname);

    if (!(endsWith(lower, ".tif") || endsWith(lower, ".tiff"))) continue;

    if (endsWith(lower, "_0000.tif") || endsWith(lower, "_0000.tiff")) {
        print("skip (already _0000): " + fname);
        skipped++;
        continue;
    }

    dot  = lastIndexOf(fname, ".");
    base = substring(fname, 0, dot);
    ext  = substring(fname, dot);                 // ".tif" or ".tiff"
    newname = base + "_0000" + ext;

    ok = File.rename(images_dir + sep + fname, images_dir + sep + newname);
    if (ok) {
        n++;
        print("[" + n + "]  " + fname + "  ->  " + newname);
    } else {
        print("ERROR renaming: " + fname);
    }
}

print("");
print("Done.  Added _0000 to " + n + " image(s).  Skipped " + skipped +
      " already-suffixed.  (Labels in labelsTr/ left unchanged.)");
