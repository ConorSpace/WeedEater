import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/conor/Developer/school/capstone/WeedEater/rosws/install/controller'
