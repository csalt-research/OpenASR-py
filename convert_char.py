import sys

def convert_to_char_seq(filepath):
	lines = open(filepath, 'r').readlines()
	lines = [l.strip().replace(' ', '_').lower() for l in lines]
	lines = [' '.join(list(l))+'\n' for l in lines]
	f = open(filepath + '.converted', 'w')
	f.writelines(lines)
	f.close()

if __name__ == '__main__':
	convert_to_char_seq(sys.argv[1])