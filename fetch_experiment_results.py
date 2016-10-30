import getpass
# import ipdb
# print ("config floatX: {}".format(config.floatX))
import shutil
import os
import ipdb

username = getpass.getuser()
if username == "sylvaint" and not os.path.isdir("/Tmp/sylvaint"):
    os.makedirs("/Tmp/sylvaint")


def copy_results(original_path, destination_path):
    ignore_pattern = shutil.ignore_patterns("model*")
    # ipdb.set_trace()
    for dirname, dirnames, filenames in os.walk(original_path):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            if not subdirname.startswith("learn_gene"):
                continue

            #if os.path.join(destination_path, subdirname)

            if os.path.exists(os.path.join(destination_path, subdirname)):
                shutil.rmtree(os.path.join(destination_path, subdirname))
            print "Copying: {}".format(subdirname)
            shutil.copytree(
                    os.path.join(dirname, subdirname),
                    os.path.join(destination_path, subdirname),
                    symlinks=False,
                    ignore=ignore_pattern)

        print "Copied total of {} directories.".format(len(dirnames))

if __name__ == '__main__':
    original_path = "/Tmp/sylvaint/feature_selection/"
    destination_path = "/data/lisatmp4/sylvaint/" + \
        "feature_selection/feature_selection"

    copy_results(original_path, destination_path)
