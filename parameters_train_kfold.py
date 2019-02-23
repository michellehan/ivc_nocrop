import sys

if len(sys.argv) == 2:
    flag = str(sys.argv[1])
train_csv = '/data/IVC_180814/IVC_Filter_Images/labels/train_kfold_' + str(flag) + '.csv'
val_csv = '/data/IVC_180814/IVC_Filter_Images/labels/val_kfold_' + str(flag) + '.csv'

arguments = ""
parameters = {
	######## dataset #######
	"train-dir"	    : "/data/IVC_180814/IVC_Filter_Images/train",
	"val-dir" 	    : "/data/IVC_180814/IVC_Filter_Images/train",
	"train-csv" 	    : train_csv,
	"val-csv" 	    : val_csv,

	####### model variables #######
	"pretrained"        : 1, 			# use pretrained dataaet 
	"dataset"           : "imagenet", 		# choose pretrained format: imagetnet, cifar10, cxr14
	"arch"              : "resnet18",
	"final-model"       : "primary", 		# primary or ema

	"batch-size"        : 32,
	"labeled-batch-size": 32,
	"start-epoch"	    : 0,
	"epochs" 	    : 30,
	"evaluation-epochs" : 3,			# how often do you check in during training
	"checkpoint-epochs" : 3,			# how often do save a checkpoint during training

	"num-classes"       : 14,
	"lr" 		    : 0.1,
	"lr-decay"          : 25,                       # lr *= 0.25 every n epochs
	"momentum" 	    : 0.85,
	"weight-decay" 	    : 0.0001,
	"ema-decay" 	    : 0.999,
	"nesterov" 	    : False,			# use nesterov momentum
	"consistency" 	    : 0, 			# use consistency loss with given weight


	####### running parameters #######
	"flag"              : str(flag), 		# full, balanced, or unbalanced training
	"evaluate"          : 0, 			# evaluate model? 0 or 1; if =1, log must be turned off
 	"resume" 	    : 0,                        
 	"ckpt" 		    : "best",                   # specify best or final

	"log"               : 1, 			# log to text file
	"print-freq" 	    : 10, 		    	# console print progress for every n batches


	####### GPU parameters #######
	"seed"              : 2,
	"workers" 	    : 4,
}

for key, value in parameters.items(): arguments += "--" + str(key) + " "+ str(value) + " "
print(arguments)
