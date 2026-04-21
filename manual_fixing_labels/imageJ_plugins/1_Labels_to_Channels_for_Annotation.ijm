// Initialization
requires("1.53c");
run("Options...", "iterations=3 count=1 black edm=Overwrite");
run("Colors...", "foreground=white background=black selection=yellow");
run("Clear Results"); 
run("Close All");

// Parameters
rawpath = getDirectory("select folder containing imagesTr and labelsTr subfolders");
MaxLabel = getNumber("Maximum label intensity:", 6);
//MIPon = false;
//TIFon = false;

start = getTime();
setBatchMode(true);

print("\\Clear");
print(rawpath);
setOption("BlackBackground", true);
images_dir = rawpath + "/imagesTr/";
labels_dir = rawpath + "/labelsTr/";

filelist = getFileList(images_dir);

output = rawpath + "/labels_as_channels/";

File.makeDirectory(output);

for (i = 0; i < lengthOf(filelist); i++) {
    if (endsWith(filelist[i], ".tif")) { 
    		
	    print("\\Update1:  Processing file " + filelist[i] + ". File " + (i+1) + " of " + filelist.length + ".");

		// open image
		open(images_dir + filelist[i]);
		rename("Raw");
		run("8-bit");
		
		labelname = filelist[i];
		open(labels_dir + labelname);
		rename("labels");
		
		for (j = 1; j <= MaxLabel; j++) {
			selectWindow("labels");
			run("Duplicate...", "title=" + j + " duplicate");
			setAutoThreshold("Default dark no-reset");
			setThreshold(j, j, "raw");
			run("Convert to Mask", "background=Dark black");
		}

		if (MaxLabel == 2) run("Merge Channels...", "c1=Raw c2=1 c3=2 create");
		if (MaxLabel == 3) run("Merge Channels...", "c1=Raw c2=1 c3=2 c4=3 create");	
		if (MaxLabel == 4) run("Merge Channels...", "c1=Raw c2=1 c3=2 c4=3 c5=4 create");
		if (MaxLabel == 5) run("Merge Channels...", "c1=Raw c2=1 c3=2 c4=3 c5=4 c6=5 create");
		if (MaxLabel == 6) run("Merge Channels...", "c1=Raw c2=1 c3=2 c4=3 c5=4 c6=5 c7=6 create");
		
		save(output + labelname);
		
		close("*");
		run("Collect Garbage");
	}
}

end = getTime();
time = (end-start)/1000/60;
print("Processing time =", time, "minutes");			

function ImageFilesOnlyArray(arr) {
	//pass array from getFileList through this e.g. NEWARRAY = ImageFilesOnlyArray(NEWARRAY);
	setOption("ExpandableArrays", true);
	f = 0;
	files = newArray;
	for (i = 0; i < arr.length; i++) {
		if (endsWith(arr[i], ".tif") || endsWith(arr[i], ".nd2") || endsWith(arr[i], ".LSM") || endsWith(arr[i], ".czi") || endsWith(arr[i], ".jpg")) {
			files[f] = arr[i];
			f = f + 1;
		}
	}
	arr = files;
	arr = Array.sort(arr);
	return arr;
}

function short_title(imagename) {
	nl = lengthOf(imagename);
	nl2 = nl - 4;
	Sub_Title = substring(imagename, 0, nl2);
	Sub_Title = replace(Sub_Title, "(", "_");
	Sub_Title = replace(Sub_Title, ")", "_");
	Sub_Title = replace(Sub_Title, "-", "_");
	Sub_Title = replace(Sub_Title, "+", "_");
	Sub_Title = replace(Sub_Title, " ", "_");
	Sub_Title = Sub_Title + ".tif";
	return Sub_Title;
}