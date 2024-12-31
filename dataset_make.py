import shutil
import cv2
import os

def txt_translate(path, txt_path):
    print(path)
    print(txt_path)
    # Make sure the path exists
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    for filename in os.listdir(path):
        # print(filename)
        list1 = filename.split("-", 3)  # split using "-"
        subname = list1[2]
        list2 = filename.split(".", 1)
        subname1 = list2[1]
        if subname1 == 'txt':
            continue
        lt, rb = subname.split("_", 1)  # split using "_"
        lx, ly = lt.split("&", 1)
        rx, ry = rb.split("&", 1)
        width = int(rx) - int(lx)
        height = int(ry) - int(ly)  # bounding box width & height
        cx = float(lx) + width / 2
        cy = float(ly) + height / 2  # bounding box central point

        img = cv2.imread(os.path.join(path, filename))
        if img is None:  # delete broken png files
            print(path + filename)
            os.remove(path + filename)
            continue
        width = width / img.shape[1]  # Normalization
        height = height / img.shape[0]
        cx = cx / img.shape[1]
        cy = cy / img.shape[0]

        txtname = filename.split(".", 1)
        txtfile = os.path.join(txt_path, txtname[0] + ".txt")
        # Green plate - class 0
        with open(txtfile, "w") as f:
            f.write(str(0) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))


if __name__ == '__main__':
    # det image
    trainDir = r"./CCPD2020/ccpd_green/train/"
    validDir = r"./CCPD2020/ccpd_green/val/"
    testDir = r"./CCPD2020/ccpd_green/test/"
    # det txt
    train_txt_path = r"./CCPD2020/ccpd_green/train_labels/"
    val_txt_path = r"./CCPD2020/ccpd_green/val_labels/"
    test_txt_path = r"./CCPD2020/ccpd_green/test_labels/"
    txt_translate(trainDir, train_txt_path)
    txt_translate(validDir, val_txt_path)
    txt_translate(testDir, test_txt_path)
