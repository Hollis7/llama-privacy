import random,json

def write_txt(file_path,datas):
    with open(file_path,"w",encoding="utf8") as f:
        for d in datas:
            f.write(json.dumps(d,ensure_ascii=False)+"\n")
        f.close()

with open("/home/user/hdb/projects/pyProject/llama-privacy/data-tmp/train_0.5M_CN/Belle_open_source_0.5M.json","r",encoding="utf8") as f:
    lines=f.readlines()
    #拼接数据
    changed_data=[]
    for l in lines:
        l=json.loads(l)
        changed_data.append({"text":"### Human: "+l["instruction"]+" ### Assistant: "+l["output"]})

    #从拼好后的数据中，随机选出1000条，作为训练数据
    #为了省钱 和 演示使用，我们只用1000条，生产环境至少要使用全部50w条
    r_changed_data=random.sample(changed_data, 1000)

    #写到json中
    write_txt("/home/user/hdb/projects/pyProject/llama-privacy/data-tmp/Belle_open_source_0.5M_changed_test2.json",r_changed_data)
