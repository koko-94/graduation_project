# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cangyi/桌面/pointcloud/graduation_project/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cangyi/桌面/pointcloud/graduation_project/src/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/main.cpp.o: ../main.cpp
CMakeFiles/main.dir/main.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cangyi/桌面/pointcloud/graduation_project/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/main.cpp.o -MF CMakeFiles/main.dir/main.cpp.o.d -o CMakeFiles/main.dir/main.cpp.o -c /home/cangyi/桌面/pointcloud/graduation_project/src/main.cpp

CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cangyi/桌面/pointcloud/graduation_project/src/main.cpp > CMakeFiles/main.dir/main.cpp.i

CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cangyi/桌面/pointcloud/graduation_project/src/main.cpp -o CMakeFiles/main.dir/main.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/main.cpp.o
main: CMakeFiles/main.dir/build.make
main: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
main: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
main: /usr/lib/x86_64-linux-gnu/libpcl_people.so
main: /usr/lib/libOpenNI.so
main: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
main: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
main: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
main: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
main: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
main: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
main: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
main: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
main: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
main: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
main: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
main: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
main: /usr/lib/x86_64-linux-gnu/libpcl_features.so
main: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
main: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
main: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
main: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
main: /usr/lib/x86_64-linux-gnu/libpcl_search.so
main: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
main: /usr/lib/x86_64-linux-gnu/libpcl_io.so
main: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
main: /usr/lib/x86_64-linux-gnu/libpng.so
main: /usr/lib/x86_64-linux-gnu/libz.so
main: /usr/lib/libOpenNI.so
main: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
main: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
main: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
main: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkIOCore-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libfreetype.so
main: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkIOImage-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkRenderingUI-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libvtkkissfft-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libGLEW.so
main: /usr/lib/x86_64-linux-gnu/libX11.so
main: /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so.5.15.3
main: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.15.3
main: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.15.3
main: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.15.3
main: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libtbb.so.12.5
main: /usr/lib/x86_64-linux-gnu/libvtksys-9.1.so.9.1.0
main: /usr/lib/x86_64-linux-gnu/libpcl_common.so
main: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
main: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.74.0
main: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.74.0
main: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so.1.74.0
main: /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.74.0
main: /usr/lib/x86_64-linux-gnu/libqhull_r.so.8.0.2
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cangyi/桌面/pointcloud/graduation_project/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /home/cangyi/桌面/pointcloud/graduation_project/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cangyi/桌面/pointcloud/graduation_project/src /home/cangyi/桌面/pointcloud/graduation_project/src /home/cangyi/桌面/pointcloud/graduation_project/src/build /home/cangyi/桌面/pointcloud/graduation_project/src/build /home/cangyi/桌面/pointcloud/graduation_project/src/build/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

