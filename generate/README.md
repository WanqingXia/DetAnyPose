# Generate iamges with BlenderProc
1. Run calc_box to calculate the object's bounding box diagonal length with python calc_box.py.
2. Run generate.py to sample camera points and generate images with "blenderproc run generate.py".
If the previous command does not work, try python ./../cli.py run generate.py
3. (Optional) Run sample_sphere to visualise the camera points sampled around object.
4. (Optional) Run search.py to search the closest image given an real image and mat file.