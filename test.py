import pandas as pd

# 创建一个空的DataFrame
acc_df = pd.DataFrame(columns=['Epoch', 'Accuracy'])

# 添加数据
acc_df = acc_df.append({'Epoch': 1, 'Accuracy': 0.9}, ignore_index=True)

# 打印结果，看看是否成功添加
print(acc_df)