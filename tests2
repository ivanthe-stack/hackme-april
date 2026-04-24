this sesion was about changing archs 
1 [64, 256, 64, 16, 10]
/loss-graphs/
loss_e_1000_lr_0.000001_adam_arch_64_256_64_16_10_seed_123.png
loss_e_2500_lr_0.000001_adam_arch_64_256_64_16_10_seed_123.png
loss_e_5000_lr_0.000001_adam_arch_64_256_64_16_10_seed_123.png
this arch has worse minmimal loss than [64, 16, 10]
but training is way more stable but this might be just because porpotionaly lr is smaler
a few more tests
/loss-graphs/
loss_e_500_lr_00_lr_0.00001_adam_arch_64_256_64_16_10_seed_123.png
loss_e_50_lr_0.0001_adam_arch_64_256_64_16_10_seed_123.png
this loss curves are esentaly the same as 
loss_e_5000_lr_0.000001_adam_arch_64_256_64_16_10_seed_123.png
but 100x less epochs and 100x higher learning rate
/loss-graphs/
loss_e_500_lr_0.0001_adam_arch_64_256_10_seed_123.png
i dicided that maybe having only one input layer will be interesting
it was better
Test Loss: 0.5802
Test Accuracy: 0.1361
i just found out after seeing how small the acc is that something is not corect
it was a critical bug in train.py in the SGD using older copied weights instead of the corect ones and i decide that i will not be using adam for now
/loss-graphs/
loss_e_500_lr_0.0001_sgd_arch_64_256_10_seed_123.png
now it is better
