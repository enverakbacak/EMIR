function score = EMRscore(H ,A, y0)
    tmp = H'*y0;
    tmp = A\tmp;
    score = y0 - H*tmp;

end



