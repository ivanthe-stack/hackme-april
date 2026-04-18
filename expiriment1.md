at first the project had issues for perfoirmance that is why we made our own port of autograd that is in numpy instead of stdlib the https://github.com/TUES-AI/AutoGrad repo 
then there were issues with -0.00000(NaN) loss i found out how floats actualy work and i belive that the eponte leaks over to the sign bit and it becomes negative to prevent this i added gradient cliping
after all of this during testsing i realized changung vars all to time is annoying and i adde all the flags
after seeing that the loss keeps going up i split the columns that are printed to be all the training loss testloss  and testacc and added csv so i can analyze the data after the training run
after the analyzing i realized need to add something i stoped at weight decay and after a while of texting i ditcged it for adam 


 
