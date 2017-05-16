# Recommender system for spotify user based on listened tracks
# RecSys Challenge Information Retrieval 2017
# Team: rcmmndrs

#usage
#python <recsys.py> -i <infile> -n <nrrecs> -o <outfile>
#python recsys.py -i ir2017.csv -n 10 -o results.csv

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-i", help="Name of the infile")
parser.add_argument("-n", help="Number of recommended songs per user")
parser.add_argument("-o", help="Name of the outfile")

# Read the arguments and
args = parser.parse_args()
if args.i == None:
	infile = 'ir2017.csv'
else:
	infile = args.i
if args.n == None:
	args.n = 10
else:
	n = int(args.n)
if args.o == None:
	outfile = 'results.csv'		
else:
	outfile = args.o

# Name of the infile
#print(infile)
# Number of tracks to recommend per user
#print(n)
# Name of the outfile
#print(outfile)



# Todo:
# Do the recommendation and create the result file with name outfile
print("Recommendations written in "+outfile)

