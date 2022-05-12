import pickle
import sys
import os
cnt = int(sys.argv[1])
if not os.path.exists("tmp"):
    os.mkdir("tmp")
pickle.dump(cnt, open("tmp/cnt.pk", "wb"))
