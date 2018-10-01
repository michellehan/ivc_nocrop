arguments = ""
parameters = {
	######## dataset #######
	"train-dir"	    : "/data/IVC_180814/IVC_Filter_Images/train",
	"val-dir" 	    : "/data/IVC_180814/IVC_Filter_Images/test",
	#"train-csv" 	    : "/data/IVC_180814/IVC_Filter_Images/labels/train_labels.csv", #14 classes
	#"val-csv" 	    : "/data/IVC_180814/IVC_Filter_Images/labels/test_labels.csv", 
	#"train-csv" 	    : "/data/IVC_180814/IVC_Filter_Images/labels/trainGroupedB_labels.csv", #12 classes
	#"val-csv" 	    : "/data/IVC_180814/IVC_Filter_Images/labels/testGroupedB_labels.csv", 
	#"train-csv" 	    : "/data/IVC_180814/IVC_Filter_Images/labels/trainGroupedBC_labels.csv", #11 classes
	#"val-csv" 	    : "/data/IVC_180814/IVC_Filter_Images/labels/testGroupedBC_labels.csv",
	#"train-csv" 	    : "/data/IVC_180814/IVC_Filter_Images/labels/trainGroupedBG_labels.csv", #11 classes
	#"val-csv" 	    : "/data/IVC_180814/IVC_Filter_Images/labels/testGroupedBG_labels.csv",
	"train-csv" 	    : "/data/IVC_180814/IVC_Filter_Images/labels/trainGroupedAll_labels.csv", #10 classes
	"val-csv" 	    : "/data/IVC_180814/IVC_Filter_Images/labels/testGroupedAll_labels.csv",


	####### model variables #######
	"pretrained"        : 1, 			# use pretrained dataaet 
	"dataset"           : "imagenet", 		# choose pretrained format: imagetnet, cifar10, cxr14
	"arch"              : "resnet50",
	"final-model"       : "primary", 		# primary or ema

	"batch-size"        : 32,
	"labeled-batch-size": 32,
	"start-epoch"	    : 0,
	"epochs" 	    : 100,
	"evaluation-epochs" : 5,			# how often do you check in during training
	"checkpoint-epochs" : 5,			# how often do save a checkpoint during training

	"num-classes"       : 10,
	"lr" 		    : 0.1,
	"lr-decay"          : 25,                       # lr *= 0.25 every n epochs
	"momentum" 	    : 0.85,
	"weight-decay" 	    : 0.0001,
	"ema-decay" 	    : 0.999,
	"nesterov" 	    : False,			# use nesterov momentum
	"consistency" 	    : 0, 			# use consistency loss with given weight


	####### running parameters #######
	"flag"              : "full", 			# full, balanced, or unbalanced training
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
