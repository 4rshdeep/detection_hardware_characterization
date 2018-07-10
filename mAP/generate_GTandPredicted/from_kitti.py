import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False,
	help="path to kitti labels", default="/home/arshdeep/datasets/kitti/training/label/")
args = vars(ap.parse_args())

PATH=args["path"]
cwd = os.getcwd()
if not os.path.exists("ground-truth"):
	os.makedirs("ground-truth")

os.chdir(PATH)
files = os.listdir()

gt_path = cwd + "/ground-truth/"

to_req_labels = {}
to_req_labels["Car"]             = "vehicle"
to_req_labels["Van"]             = "vehicle"
to_req_labels["Truck"]           = "vehicle"
to_req_labels["Cyclist"]         = "vehicle"
to_req_labels["Tram"]	         = "vehicle"
to_req_labels["Pedestrian"]	     = "person"
to_req_labels["Person_sitting"]	 = "person"

for file in files:
	with open(PATH+file, "r") as fr:
		with open(gt_path+file, "w") as fw:
			for line in fr.readlines():
				words = line.strip().split(" ")
				# if file == "000008.txt":
					# print(words)
				if words[0] in list(to_req_labels.keys()):
					fw.write(to_req_labels[words[0]]+" "+words[4]+" "+words[5]+" "+words[6]+" "+words[7]+"\n")
				