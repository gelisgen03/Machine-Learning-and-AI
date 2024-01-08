

word_list = ['percussion',
             'supersonic',
             'car',
             'tree',
             'boy',
             'girl',
             'arc']
score=0
list=[]
def find_anagram_words(word_list):
    while(True):
        for n in range(0,len(word_list)):
            for m in range(0,len(word_list)):
                if len(word_list[m])==len(word_list[n]) and word_list[n]!=word_list[m]:    
                    for x in range(0,len(word_list[m])):
                        for y in range(0,len(word_list[m])):
                            if word_list[n][x]==word_list[m][y]:
                                score=+1
                    if score==len(word_list[n]):            
                        list.append(word_list[n])
                        list.append(word_list[m])
        break 
find_anagram_words(word_list)
print(list)
