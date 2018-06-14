@echo off
set name=
set data_path=.
set file_type=.jpg
set save_path=D:\DataSet\tmp\
set save_file_type=.txt

mkdir %save_path%

for %%* in (%data_path%) do set name=%%~n*

set save=%save_path%%name%%save_file_type%

for %%* in (%data_path%) do DIR /B /ON *%file_type% > %save%

echo File %name%%save_file_type% at %save_path% is created

pause