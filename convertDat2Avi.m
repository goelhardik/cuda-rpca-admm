function convertDat2Avi(filename, saveFilename, h, w, fr, m, n)

fid = fopen(filename,'r');
X = fread(fid,[m,n],'float'); 
fclose(fid); 
%X = X';
X = single(X);

% above can be a problem as the output may not contain the matrix dimensions as 3 times no. of frames.

% call lrslibary.utils.convert_video2d_to_avi
convert_video2d_to_avi( X, fr, h, w, saveFilename);
