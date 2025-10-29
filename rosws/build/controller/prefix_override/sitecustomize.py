import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/pass_is_queens/Developer/WeedEater/rosws/install/controller'
