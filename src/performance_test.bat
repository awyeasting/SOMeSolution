SET OMP_NUM_THREADS=2
FOR /L %%i IN (1,1,10) DO (
	main.exe 10 10 --g 100 10 -e 100
	)