# Script

`run_file_with_ncu.sh`: Compile the code, and use nsight compute to profile kernel speed

`nvcc_compile.py`: Batch compile and output the result to a specific file

`process_data.py`: Process `nvcc_compile.py`'s output，and aggregate the results into multiple csv files

`draw.py`: Drawing according to the content of the csv file
