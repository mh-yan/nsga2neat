import numpy as np
import openpyxl
def arr_to_excel(data,sheet):
    for row in data:
        sheet.append(row)
def list_to_excel(data,sheet):
    for item in data:
        sheet.append([item])

# 格式化输出--excel
def dim2_ouput_format_excel(index_x_y_cat,Tri,path):
    """
    get excel's file

    Args:
        index_x_y_cat (_type_): numpy array of four columns
        Tri (_type_): triangulation's results
    """
    # Todo  下标从1开始
    part1_points=index_x_y_cat[:,0:-1].tolist()
    part1_points=np.array(part1_points)
    part1_points[:,0]+=1
    print(part1_points.shape[0])
    z=np.zeros((part1_points.shape[0],1))
    # Todo 加一个z坐标
    part1_points=np.hstack((part1_points,z))
    part2_tri=[]
    # Todo 下标从1开始
    for i,tri in enumerate(Tri):
        new_tri=np.insert(tri, 0, i+1, axis=0)
        new_tri[1:] += 1
        new_tri=list(map(int, new_tri))
        part2_tri.append(new_tri)
    # if_inside=index_x_y_cat[:,-1]==2
    # if_outside=index_x_y_cat[:,-1]==1
    # if_border=index_x_y_cat[:,-1]==0
    # inside_p=index_x_y_cat[if_inside][:,0].tolist()
    # outside_p=index_x_y_cat[if_outside][:,0].tolist()
    # border_p=index_x_y_cat[if_border][:,0].tolist()
    fixed_node=[]
    load_node=[]
    # define the load and support 
    for data in index_x_y_cat:
        if data[1]==3:
            if data[2]<=-0.8:
                 load_node.append(int(data[0]))
    for data in  index_x_y_cat:
        if data[1]==-3:
            fixed_node.append(int(data[0]))
            
    load_node=[i+1 for i in load_node]
    fixed_node=[i+1 for i in fixed_node]
    part1_title=[2,3,index_x_y_cat.shape[0],Tri.shape[0]]
    # part2_title=[7,'inside',int(len(inside_p))]
    # part3_title=[7,'outside',int(len(outside_p))]
    # part4_title=[7,'border',int(len(border_p))]
    part_fixed=[7,'fixed_node',int(len(fixed_node))]
    part_load=[7,'load',int(len(load_node))]
    inside_tri=[]
    outside_tri=[]
    for i,tri in enumerate(Tri):
        flag=2
        index_tri=i+1
        for node in tri:
            # Todo 下标回去
            if(index_x_y_cat[int(node),-1]==1):
                outside_tri.append(index_tri)
                flag=1
                break
        if flag==2:
            inside_tri.append(index_tri)
    
    
    part5_title=[8,"inside_tri",int(len(inside_tri))]
    part6_title=[8,"outside_tri",int(len(outside_tri))]
    # 创建一个新的 Excel 工作簿
    workbook = openpyxl.Workbook()
    # 选择默认的工作表
    sheet = workbook.active
    sheet.append(part1_title)
    arr_to_excel(part1_points.tolist(),sheet)
    arr_to_excel(part2_tri,sheet)
    # sheet.append(part2_title)
    # list_to_excel(inside_p,sheet)
    # sheet.append(part3_title)
    # list_to_excel(outside_p,sheet)
    # sheet.append(part4_title)
    # list_to_excel(border_p,sheet)
    sheet.append(part_fixed)
    list_to_excel(fixed_node, sheet)
    sheet.append(part_load)
    list_to_excel(load_node, sheet)
    sheet.append(part5_title)
    list_to_excel(inside_tri,sheet)
    sheet.append(part6_title)
    list_to_excel(outside_tri,sheet)
    excel2txt(sheet,path)
    correct_txt(path)

def excel2txt(sheet,path):
    txt_file = f"{path}/output.txt"  # 替换为您想要保存的文本文件路径
    with open(txt_file, "w") as txt_file:
        for row in sheet.iter_rows(values_only=True):
            row_values = [str(int(value)) if i==0  else str(value) for i,value in enumerate(row)]
            txt_file.write(' '.join(row_values) + '\n')
            
def correct_txt(path):
    txt_file = f"{path}/output.txt"  # 替换为您想要保存的文本文件路径
    dat_file = f"{path}/output.dat"
    old_txt=' None'
    new_txt=''
    with open(txt_file, "r") as tf:
        file_content = tf.read()
    
    modified_content = file_content.replace(old_txt, new_txt)
    
    with open(dat_file, "w") as tf:
        tf.write(modified_content)
