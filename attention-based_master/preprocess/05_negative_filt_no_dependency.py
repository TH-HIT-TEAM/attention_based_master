import re


def filter(r_path, w_path):

    ot = w_path
    of = open(ot,'w')
    
    t = r_path
    f = open(t,'r')
    print ('running %s ...'%t)
    data = f.read().strip().split('\n\n')
    c = 0
    ra = []
    rf = []
    for d in data:
        sent_blind, sent_blind_pos, relation, sent_id, sent, entities = d.strip().split('\n')
        ra.append(relation)
    
        r1 = re.search(r'DRUG_1 (, |\s )*DRUG_2 (and |or )+', sent_blind, re.I)  # DRUG_1 , DRUG_2 , and DRUG_N .
        r2 = re.search(r'DRUG_1 (\(|\[) (DRUG_N , )*DRUG_2( , DRUG_N)*', sent_blind, re.I) # DRUG_1 ( DRUG_N , DRUG_N , DRUG_2 ) or DRUG_N
        r3 = re.search(r'DRUG_1 such as (DRUG_N , )*(and |or )*DRUG_2', sent_blind, re.I) # DRUG_1 such as DRUG_N , DRUG_N , and DRUG_2 .
        r4 = re.search(r'such as DRUG_1 , (DRUG_N , )*(and |or )*DRUG_2', sent_blind, re.I) # such as DRUG_1 , DRUG_N , DRUG_2 , DRUG_N , DRUG_N ,
        r5 = re.search(r'e.g . (, )*DRUG_1 , (DRUG_N , )*(and |or )*DRUG_2', sent_blind, re.I)  # e.g . , DRUG_1 , DRUG_N , DRUG_2 , DRUG_N , DRUG_N and nefazadon
        r6 = re.search(r'DRUG_1 (, )*(DRUG_N , )*(or )*DRUG_2( , DRUG_N)*( or )*', sent_blind, re.I) # DRUG_N or DRUG_N
        r7 = re.search(r'DRUG_1 (DRUG_N )*(\[ )+(DRUG_N )*DRUG_2', sent_blind, re.I) # DRUG_1 [ DRUG_2 ]
        r8 = re.search(r'DRUG_1 ; (DRUG_N ; )*(DRUG_N \* ; )*(DRUG_N (dg)* ; )*DRUG_2', sent_blind, re.I)
        r9 = re.search(r'DRUG_1 \* ; (DRUG_N ; )*(DRUG_N \* ; )*(DRUG_N (dg)* ; )*DRUG_2', sent_blind, re.I)


        # r1 = re.search(r'DRUG_1 ( )*DRUG_2 (and |or )+', sent_blind, re.I)  # DRUG_1 , DRUG_2 , and DRUG_N .
        # r2 = re.search(r'DRUG_1 (\(|\[) (DRUG_N )*DRUG_2( DRUG_N)*', sent_blind, re.I)  # DRUG_1 ( DRUG_N , DRUG_N , DRUG_2 ) or DRUG_N
        # r3 = re.search(r'DRUG_1 such as (DRUG_N )*(and |or )*DRUG_2', sent_blind, re.I)  # DRUG_1 such as DRUG_N , DRUG_N , and DRUG_2 .
        # r4 = re.search(r'such as DRUG_1 (DRUG_N )*(and |or )*DRUG_2', sent_blind, re.I)  # such as DRUG_1 , DRUG_N , DRUG_2 , DRUG_N , DRUG_N ,
        # r5 = re.search(r'e.g . ( )*DRUG_1 (DRUG_N )*(and |or )*DRUG_2', sent_blind, re.I)  # e.g . , DRUG_1 , DRUG_N , DRUG_2 , DRUG_N , DRUG_N and nefazadon
        # r6 = re.search(r'DRUG_1 ( )*(DRUG_N )*(or )*DRUG_2( DRUG_N)*( or )*', sent_blind, re.I)  # DRUG_N or DRUG_N
        # r7 = re.search(r'DRUG_1 (DRUG_N )*(\[ )+(DRUG_N )*DRUG_2', sent_blind, re.I)  # DRUG_1 [ DRUG_2 ]
        # r8 = re.search(r'DRUG_1 ; (DRUG_N )*(DRUG_N \* )*(DRUG_N (dg)* )*DRUG_2', sent_blind, re.I)
        # r9 = re.search(r'DRUG_1 \* (DRUG_N )*(DRUG_N \* )*(DRUG_N (dg)* )*DRUG_2', sent_blind, re.I)

    
    
        # pf = 1
        # rule = r1
        # if rule and pf == 1:
        #     c = c + 1
        #     print sent_blind
        #     print relation
        #     print '----------------'
        #
        # if rule and relation!='false' and pf==2:
        #     c = c + 1
        #     print sent_blind
        #     print relation
        #     print '----------------'
    

    
        e1 = entities.split('\t')[0]
        e2 = entities.split('\t')[2]
    
    
        if not r1 and not r2 and not r3 and not r4 and not r5 and not r6 and not r7 and not r8 and not r9 and e1!=e2:
            rf.append(relation)
            of.write(sent_blind.strip() + '\n')
            of.write(sent_blind_pos.strip() + '\n')
            of.write(relation + '\n')
            of.write(sent_id + '\n')
            of.write(sent.strip() + '\n')
            of.write(entities.strip() + '\n')
            of.write('\n')

    # print c
    f.close()
    of.close()
    print ("all", "\t", "false", "\t", "mechan", "\t", "effect", "\t", "advise", "\t", "int")
    print (len(ra), "\t", ra.count('false'), "\t", ra.count('mechanism'), "\t", ra.count('effect'), "\t", ra.count('advise'), "\t", ra.count('int'))
    print (len(rf), "\t", rf.count('false'), "\t", rf.count('mechanism'), "\t", rf.count('effect'), "\t", rf.count('advise'), "\t", rf.count('int'))
    print ("----------------------------")


# execute
filter("./ddi_corpus/03pairwithsent/train_data.txt", "./ddi_corpus/05negativefilt_no_dependency/train_data.txt")
filter("./ddi_corpus/03pairwithsent/test_data.txt", "./ddi_corpus/05negativefilt_no_dependency/test_data.txt")

