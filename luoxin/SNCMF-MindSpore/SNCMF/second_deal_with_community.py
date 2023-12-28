# 打开输入文本文件和输出文本文件
with open('./dataset/friendster/friendster.community.txt', 'r') as input_file, open('community', 'w') as output_file:
    lines = input_file.readlines()

    # 用于存储数字与新ID之间的映射
    num_to_new_id = {}

    # 遍历每一行并提取唯一的数字
    unique_nums = set()
    for line in lines:
        nums = line.strip().split()
        unique_nums.update(nums)

    # 按照数字的大小排序
    sorted_nums = sorted(map(int, unique_nums))

    # 为每个数字分配新的ID
    for new_id, num in enumerate(sorted_nums):
        num_to_new_id[num] = new_id

    # 重新遍历每一行并将数字映射为新ID，并将结果写入输出文件
    for line in lines:
        nums = line.strip().split()
        mapped_nums = [str(num_to_new_id[int(num)]) for num in nums]
        output_line = ' '.join(mapped_nums) + '\n'
        output_file.write(output_line)

# 打印映射结果
print("映射结果:")
for num, new_id in num_to_new_id.items():
    print(f"数字 {num} 映射为新ID {new_id}")

print("结果已保存到community.txt文件。")




