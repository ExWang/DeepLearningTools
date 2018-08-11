from nltk.parse import stanford

model_path = "tools/stanford-parser-full-2015-04-20/englishPCFG.ser.gz"
path_to_jar = 'tools/stanford-parser-full-2015-04-20/stanford-parser.jar'
path_to_models_jar = 'tools/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar'


def getProName(node):
    return node.unicode_repr().split(" ")[0]


def getProNNstr(node):
    return node.unicode_repr().split(" ")[-1][1:-1]


def myFilter(sentence_nltk):
    root = sentence_nltk[0]
    label_list = []

    rootNP = None
    rootVP = None
    ZHUYU = "NULL"
    WEIYU = "NULL"
    BINYU = "NULL"

    if (root.label() == "S") and ("VP" in [getProName(x) for x in root.productions()]):
        # print("---<Normal sentence mode>---")

        for one in root:
            if one.label() == "NP":
                rootNP = one
            elif one.label() == "VP":
                rootVP = one

        if rootVP is None:
            root = root[0]
            for one in root:
                if one.label() == "NP":
                    rootNP = one
                elif one.label() == "VP":
                    rootVP = one

        # ===============find ZHU YU===============
        flag_loop = True
        findone = None
        while (True):
            for one in rootNP:
                if "NN" in one.label():
                    # print("Found NN!")
                    ZHUYU = one
                    flag_loop = False
                    break
                if one.label() == "NP":
                    # print("Found NP!")
                    findone = one
                    break
            rootNP = findone
            if not flag_loop:
                break
        # print(ZHUYU)
        ZHUYU = ZHUYU.leaves()[0]
        # print("ZHU YU:", ZHUYU)

        # ======================find WEI YU===========
        leftChild = rootVP[0]
        rightChild = rootVP[1]
        target_VP = None
        if ("VB" in leftChild.label()) and (rightChild.label() == "VP"):
            rootNext = rightChild
            flag_loop = True
            findone = None
            while True:
                for one in rootNext:
                    if "VB" in one.label():
                        # print("Found NN!")
                        WEIYU = one
                        flag_loop = False
                        target_VP = rootNext
                        break
                    if one.label() == "VP":
                        # print("Found NP!")
                        findone = one
                        break
                rootNext = findone
                if not flag_loop:
                    break
            WEIYU = WEIYU.leaves()[0]

        elif ("VB" in leftChild.label()) and (rightChild.label() != "VP"):
            WEIYU = leftChild.leaves()[0]
            target_VP = rootVP
        else:
            print("Error! Need Debug.")
            raise IndexError
        # print("WEI YU:", WEIYU)

        # ===========================find BIN YU=========================
        target_VP_pro = target_VP.productions()
        # print(target_VP_pro)
        for one in target_VP_pro:
            if "NN" in getProName(one):
                BINYU = getProNNstr(one)
                break
        # print("BIN YU:", BINYU)

        group = [ZHUYU, BINYU, WEIYU]
        label_list.append(group)


    elif (root.label() == "NP") and ("VP" not in [getProName(x) for x in root.productions()]):
        # print("---<NP couples mode>---")
        WEIYU = "NULL"
        root_main = None
        root_sub_list = []
        flag_once = True
        for one in root:
            if one.label() == "NP" and flag_once:
                root_main = one
                flag_once = False
            elif (one.label() == "NP" or one.label() == "PP") and not flag_once:
                root_sub_list.append(one)

        # ===============find ZHU YU===============
        flag_loop = True
        findone = None
        while (True):
            for one in root_main:
                if "NN" in one.label():
                    # print("Found NN!")
                    ZHUYU = one
                    flag_loop = False
                    break
                if one.label() == "NP":
                    # print("Found NP!")
                    findone = one
                    break
            root_main = findone
            if not flag_loop:
                break
        # print(ZHUYU)
        ZHUYU = ZHUYU.leaves()[0]
        # print("ZHU YU:", ZHUYU)

        # ===============find BIN YU===============
        for one_tree in root_sub_list:
            one_pro = one_tree.productions()
            for one in one_pro:
                if "NN" in getProName(one):
                    BINYU = getProNNstr(one)
                    # print("BIN YU:", BINYU)
                    group = [ZHUYU, BINYU, WEIYU]
                    label_list.append(group)


    elif root.label() == "FRAG":
        # print("---<FRAG mode>---")
        rootNP = None
        rootS = None
        flag_once = True
        for one in root:
            if one.label() == "NP" and flag_once:
                rootNP = one
                flag_once = False
            elif one.label() == "S" and not flag_once:
                rootS = one

        # ===============find ZHU YU===============
        flag_loop = True
        findone = None
        while (True):
            for one in rootNP:
                if "NN" in one.label():
                    # print("Found NN!")
                    ZHUYU = one
                    flag_loop = False
                    break
                if one.label() == "NP":
                    # print("Found NP!")
                    findone = one
                    break
                    rootNP = findone
            if not flag_loop:
                break
        # print(ZHUYU)
        ZHUYU = ZHUYU.leaves()[0]
        # print("ZHU YU:", ZHUYU)

        # ===============find BIN YU and WEI YU===============
        S_pro = rootS.productions()
        lastVP = None
        index_lastVP = 0
        i = 0
        # print(S_pro)
        for one in S_pro:
            if "VP" in getProName(one):
                index_lastVP = i
            i += 1
        # print(lastVP, type(lastVP))
        # print(index_lastVP)
        WEIYU = getProNNstr(S_pro[index_lastVP + 1])
        i = 0
        index_target_NN = 0
        for one in S_pro:
            if ("NN" in getProName(one)) and i > index_lastVP:
                index_target_NN = i
                break
            i += 1
        BINYU = getProNNstr(S_pro[index_target_NN])

        # print("WEI YU:", WEIYU)
        # print("BIN YU:", BINYU)
        group = [ZHUYU, BINYU, WEIYU]
        label_list.append(group)

    else:
        # print("---<Other mode>---")
        index_ZHUYU = 0
        index_BINYU = 0
        index_WEIYU = 0

        i = 0
        root_pro = root.productions()
        for one in root_pro:
            if "NN" in getProName(one):
                index_ZHUYU = i
                break
            i += 1
        ZHUYU = getProNNstr(root_pro[index_ZHUYU])

        index_lastVP = 0
        i = 0
        for one in root_pro:
            if "VP" in getProName(one):
                index_lastVP = i
                index_WEIYU = index_lastVP + 1
            i += 1
        WEIYU = getProNNstr(root_pro[index_WEIYU])

        i = 0
        for one in root_pro:
            if ("NN" in getProName(one)) and i > index_lastVP:
                index_BINYU = i
                break
            i += 1
        BINYU = getProNNstr(root_pro[index_BINYU])

        # print("ZHU YU:", ZHUYU)
        # print("WEI YU:", WEIYU)
        # print("BIN YU:", BINYU)
        group = [ZHUYU, BINYU, WEIYU]
        label_list.append(group)

    return label_list


def myMain():
    parser = stanford.StanfordParser(path_to_jar=path_to_jar,
                                     path_to_models_jar=path_to_models_jar,
                                     model_path=model_path)

    test_sentences = ("A man in white shirt on bicycle with a dog riding in the back.",
                      "several young students working at a desk with multiple computers",
                      "a person running a bike on a road with trees in the background",
                      "A white tank with an book on it in classroom.",
                      "A young woman standing in a kitchen eats a plate of vegetables.",
                      "A man in a red shirt and a red hat is on a motorcycle on a hill side",
                      "Stone keepers on the ground is holding a gem of time.",
                      "Two chefs in a restaurant kitchen preparing food. ",
                      "A commercial dish washing station with a toilet in it.",
                      "A geoup of people on bicycles coming down a street.",
                      "a bathroom with a toilet between a sink and a shower",
                      "Elephant walking through the middle of the road in front of a car. ",
                      "A horse drawn carriage among several other motor vehicles on a road.",
                      "A car is parked near a parking meter.",
                      "a row of bikes and mopeds is parked along the street",
                      )

    sentences = parser.raw_parse_sents(test_sentences)

    for line in sentences:
        for sentence in line:
            # sentence.draw()
            zhuweibin = myFilter(sentence)

            # print(zhuweibin)


if __name__ == "__main__":
    myMain()
