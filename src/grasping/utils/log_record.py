"""
A log file to record the programming status
"""
def create_log(dir):
    f = open(dir + "/log.txt","w")  # create a txt file and delete the last one
    f.close()

def update_log(dir, msg):
    f = open(dir + "/log.txt","a+")  # add the log info at the end the file
    f.write("************************\n")
    f.write(msg + "\n")
    f.close()