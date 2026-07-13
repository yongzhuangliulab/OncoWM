######### section 1###########




#########  tsne ############

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

# ==========================================
# 1. 统一配置文件路径与变量名
# ==========================================
csv_path = "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/ALL_OS_dataset.csv"

FEATURE_PATHS = {
    "OncoJEPA":          "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Full_modal/full_cohort/joint_embeddings.npz",
    "OncoJEPA w/o Gen":   "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Modal_miss_geno/full_cohort/joint_embeddings.npz",
    "OncoJEPA w/o Tran":  "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Modal_miss_tran/full_cohort/joint_embeddings.npz",
    "OncoJEPA w/o Path":  "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Modal_miss_path/full_cohort/joint_embeddings.npz",
    "OncoJEPA w/o Clin":  "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Modal_miss_text/full_cohort/joint_embeddings.npz",
    "OncoJEPA Gen-Only":        "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Only_geno/full_cohort/joint_embeddings.npz",
    "OncoJEPA Tran-Only":       "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Only_tran/full_cohort/joint_embeddings.npz",
    "OncoJEPA Path-Only":       "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Only_path/full_cohort/joint_embeddings.npz",
    "OncoJEPA Clin-Only":       "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Only_text/full_cohort/joint_embeddings.npz", 
    "TITAN (Path)":        "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_SVS/TCGA_TITAN_first_slice_dict.pt",
    "Medgamma (Clin)":        "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/TCGA_data_clinical_patient_prompt.pt", 
    "SNN (Gen)":         "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_genome/TCGA-ALL.pt",
    "BulkFormer (Tran)":        "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_transcriptome/TCGA-ALL.pt", 
    'Prov-GigaPath (Path)': '/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/Prov_gigapath/provgigapath_embeddings_with_metadata.pt',
    "Concat Fusion": "/home/Disk30T/chengpeng/Multimodal_JEPA/result/Baseline_Fusions/ALL/Concat_features.pt",
    'Mean-Pool': '/home/Disk30T/chengpeng/Multimodal_JEPA/result/Baseline_Fusions/ALL/Mean_Pool_features.pt',
    'Honeybee': '/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/Honeybee_emb/final_multimodal_features.pt'
}

# ==========================================
# 2. 动态全自动解析函数
# ==========================================
def parse_feature_file(filepath):
    feat_dict = {}
    if not os.path.exists(filepath):
        print(f"   [警告] 文件路径不存在，将跳过该模态: {filepath}")
        return None
        
    if filepath.endswith(".pt"):
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            for sid, value in data.items():
                clean_sid = str(sid)[0:12].strip()
                if isinstance(value, dict):
                    if 'input_ids' in value:
                        tensor_val = value['input_ids']
                        feat = tensor_val.cpu().detach().numpy() if isinstance(tensor_val, torch.Tensor) else np.array(tensor_val)
                        feat_dict[clean_sid] = feat.reshape(-1)
                else:
                    feat = value.cpu().detach().numpy() if isinstance(value, torch.Tensor) else np.array(value)
                    feat_dict[clean_sid] = feat.reshape(-1)
        else:
            raise ValueError(f"{filepath} 不是字典格式。")
            
    elif filepath.endswith(".npz"):
        try:
            data = np.load(filepath, allow_pickle=True)
            data_dict = dict(data)
            data_dict['sample_id'] = np.array([s[:12] for s in data_dict['sample_id']])
            ids = data_dict['sample_id']
            feats = data_dict['joint_embedding']
            for sid, feat in zip(ids, feats):
                clean_sid = str(sid).strip()
                feat_dict[clean_sid] = feat.reshape(-1)
        except KeyError as e:
            print(f"   [错误] {filepath} 结构异常: {e}")
            return None
    else:
        print(f"   [警告] 无法识别的文件格式: {filepath}")
        return None
        
    return feat_dict

# ==========================================
# 3. 加载元数据 & 提取全局类别列表
# ==========================================
print("正在加载 CSV 元数据...")
df_meta = pd.read_csv(csv_path)
df_meta.iloc[:, 0] = df_meta.iloc[:, 0].astype(str).str[:12]
meta_dict = dict(zip(df_meta.iloc[:, 0], df_meta.iloc[:, 1]))

# 提取全局存在的肿瘤类型，用于统一图例颜色
GLOBAL_CANCER_TYPES = sorted(list(set(meta_dict.values())))

# ==========================================
# 4. 动态独立计算每个模态的指标与 t-SNE
# ==========================================
results = []
tsne_results_dict = {}

print(f"\n开始逐一进行特征解析、聚类与 t-SNE (每个模态独立取交集)...")
for modality_name, path in FEATURE_PATHS.items():
    feat_dict = parse_feature_file(path)
    if feat_dict is None or len(feat_dict) == 0:
        continue
    
    # 单独计算该模态与元数据的交集
    valid_ids = sorted(list(set(feat_dict.keys()) & set(meta_dict.keys())))
    
    if len(valid_ids) == 0:
        print(f"   [跳过] {modality_name} 与 CSV 无共有样本。")
        continue
        
    print(f" 处理中: {modality_name} (匹配到 {len(valid_ids)} 个样本)...")
    features = np.array([feat_dict[sid] for sid in valid_ids])
    aligned_cancer_types = np.array([meta_dict[sid] for sid in valid_ids])
    n_clusters = len(np.unique(aligned_cancer_types))
    
    # 指标计算
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(features)
    nmi = normalized_mutual_info_score(aligned_cancer_types, pred_labels)
    ami = adjusted_mutual_info_score(aligned_cancer_types, pred_labels)
    
    results.append({
        "Modality": modality_name,
        "Sample_Count": len(valid_ids),
        "NMI": nmi,
        "AMI": ami
    })
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    tsne_coords = tsne.fit_transform(features)
    
    # 保存结果以及对应的 label，画图时用
    tsne_results_dict[modality_name] = {
        "coords": tsne_coords,
        "labels": aligned_cancer_types
    }

print("\n[所有特征处理完毕]")

# ==========================================
# 辅助画图函数
# ==========================================
def plot_single_tsne(ax, modality_name, tsne_data):
    # 使用全局的 hue_order 保证各子图之间颜色对应不乱
    sns.scatterplot(
        x=tsne_data["coords"][:, 0], y=tsne_data["coords"][:, 1],
        hue=tsne_data["labels"], hue_order=GLOBAL_CANCER_TYPES, palette="tab20", 
        ax=ax, s=35, alpha=0.7, edgecolor='w', linewidth=0.3
    )
    ax.set_title(f"{modality_name}", fontsize=13, fontweight='bold')
    ax.set_xlabel("t-SNE Dim 1", fontsize=10)
    ax.set_ylabel("t-SNE Dim 2", fontsize=10)
    if ax.get_legend():
        ax.get_legend().remove()

# ==========================================
# 5. 生成 图1 (主图排版)
# ==========================================
print("\n正在生成 图1 (特定排版)...")
fig1, axes1 = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

# 按照最新要求的主图排版
fig1_layout_map = {
    "OncoJEPA": axes1[0, 0],         
    "TITAN (Path)": axes1[0, 1],       
    "OncoJEPA w/o Path": axes1[1, 0], 
    "OncoJEPA Path-Only": axes1[1, 1]  
}

handles_global, labels_global = None, None

for mod_name, ax in fig1_layout_map.items():
    if mod_name in tsne_results_dict:
        plot_single_tsne(ax, mod_name, tsne_results_dict[mod_name])
        # 尝试抓取图例句柄 (由于使用了 hue_order，任何一个图都可以提取全套图例，前提是数据够全)
        if handles_global is None and len(np.unique(tsne_results_dict[mod_name]["labels"])) > 0:
            # 利用 seaborn 自动生成的图例，抓取完即隐藏
            ax.legend(title="Cancer Type", loc='best')
            handles_global, labels_global = ax.get_legend_handles_labels()
            ax.get_legend().remove()
    else:
        ax.axis('off')
        ax.set_title(f"{mod_name} (Data Missing/Failed)")

# 处理图1的图例空间
axes1[0, 2].axis('off')
axes1[1, 2].axis('off')

if handles_global:
    axes1[0, 2].legend(
        handles_global, labels_global, 
        title="Cancer Type", loc='center', 
        fontsize=11, title_fontsize=13, ncol=2
    )

# ==========================================
# 6. 生成 图2 (画全部模态)
# ==========================================
print("正在生成 图2 (全部模态)...")
# 全部画出
fig2_keys = list(tsne_results_dict.keys())

if len(fig2_keys) > 0:
    ncols2 = 4
    nrows2 = math.ceil((len(fig2_keys) + 1) / ncols2)
    fig2, axes2 = plt.subplots(nrows2, ncols2, figsize=(5*ncols2, 4.5*nrows2), constrained_layout=True)
    axes2_flat = axes2.flatten()
    
    for idx, mod_name in enumerate(fig2_keys):
        plot_single_tsne(axes2_flat[idx], mod_name, tsne_results_dict[mod_name])
    
    # 安置 图2 的图例
    legend_idx = len(fig2_keys)
    if legend_idx < len(axes2_flat) and handles_global:
        axes2_flat[legend_idx].axis('off')
        axes2_flat[legend_idx].legend(
            handles_global, labels_global, 
            title="Cancer Type", loc='center', 
            fontsize=11, title_fontsize=13, ncol=2
        )
    
    # 删掉 图2 多余的画板
    for i in range(legend_idx + 1, len(axes2_flat)):
        fig2.delaxes(axes2_flat[i])

# ==========================================
# 7. 结果保存
# ==========================================
output_dir = "/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/"
# 为 section1 建立子文件夹，防止由于目录不存在导致保存报错
section1_dir = os.path.join(output_dir, "section1")
os.makedirs(section1_dir, exist_ok=True)

# 保存图1
img1_path = os.path.join(section1_dir, "S1_tsne_joint_pathology_classifiy_tumortype.pdf")
fig1.savefig(img1_path, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"\n[成功] 图1 (特定排版) 已保存至: {img1_path}")

# 保存图2
if len(fig2_keys) > 0:
    img2_path = os.path.join(section1_dir, "S1_tsne_joint_pathology_classifiy_tumortype_supplement_ALL.pdf")
    fig2.savefig(img2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"[成功] 图2 (全部模态) 已保存至: {img2_path}")

# 保存合并的表格
df_results = pd.DataFrame(results).set_index("Modality")
# 对表格重新按 NMI 降序一下，方便查看效果
df_results = df_results.sort_values(by="NMI", ascending=False)
csv_save_path = os.path.join(section1_dir, "S1_tsne_metrics_all_modalities.csv")
df_results.to_csv(csv_save_path)

print("\n=======================================================")
print("  总评估指标表格 (各模态依据自身有效样本计算)")
print("=======================================================")
print(df_results)
print(f"\n[成功] 聚类指标表格已保存至: {csv_save_path}")

#########  random forest ############
import numpy as np
import pandas as pd
import torch
import os
from sklearn.model_selection import StratifiedKFold  # 引入五折交叉验证工具
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==========================================
# 1. 配置路径
# ==========================================
CSV_PATH = "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/ALL_OS_dataset.csv"

FEATURE_PATHS = {
    "OncoJEPA":          "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Full_modal/full_cohort/joint_embeddings.npz",
    "OncoJEPA w/o Gen":   "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Modal_miss_geno/full_cohort/joint_embeddings.npz",
    "OncoJEPA w/o Tran":  "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Modal_miss_tran/full_cohort/joint_embeddings.npz",
    "OncoJEPA w/o Path":  "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Modal_miss_path/full_cohort/joint_embeddings.npz",
    "OncoJEPA w/o Clin":  "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Modal_miss_text/full_cohort/joint_embeddings.npz",
    "OncoJEPA Gen-Only":        "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Only_geno/full_cohort/joint_embeddings.npz",
    "OncoJEPA Tran-Only":       "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Only_tran/full_cohort/joint_embeddings.npz",
    "OncoJEPA Path-Only":       "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Only_path/full_cohort/joint_embeddings.npz",
    "OncoJEPA Clin-Only":       "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/ALL/ALL_OS/align/Only_text/full_cohort/joint_embeddings.npz", 
    "TITAN (Path)":        "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_SVS/TCGA_TITAN_first_slice_dict.pt",
    "Medgamma (Clin)":        "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/TCGA_data_clinical_patient_prompt.pt", 
    "SNN (Gen)":         "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_genome/TCGA-ALL.pt",
    "BulkFormer (Tran)":        "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_transcriptome/TCGA-ALL.pt", 
    'Prov-GigaPath (Path)': '/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/Prov_gigapath/provgigapath_embeddings_with_metadata.pt',
    "Concat Fusion": "/home/Disk30T/chengpeng/Multimodal_JEPA/result/Baseline_Fusions/ALL/Concat_features.pt",
    'Mean-Pool': '/home/Disk30T/chengpeng/Multimodal_JEPA/result/Baseline_Fusions/ALL/Mean_Pool_features.pt',
    'Honeybee': '/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/Honeybee_emb/final_multimodal_features.pt'
}

# ==========================================
# 2. 辅助解析函数 (加入截断保证匹配)
# ==========================================
def parse_feature_file(filepath):
    if not os.path.exists(filepath):
        print(f"   [警告] 文件路径不存在，将跳过: {filepath}")
        return None
    fmt = "npz" if str(filepath).endswith(".npz") else "pt"
    feat_dict = {}
    
    if fmt == "pt":
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            for sid, value in data.items():
                clean_sid = str(sid)[0:12].strip()
                if isinstance(value, dict):
                    if 'input_ids' in value:
                        tensor_val = value['input_ids']
                        feat = tensor_val.cpu().detach().numpy() if isinstance(tensor_val, torch.Tensor) else np.array(tensor_val)
                        feat_dict[clean_sid] = feat.reshape(-1)
                else:
                    feat = value.cpu().detach().numpy() if isinstance(value, torch.Tensor) else np.array(value)
                    feat_dict[clean_sid] = feat.reshape(-1)
        else:
            raise ValueError(f"{filepath} 不是字典格式。")
            
    elif fmt == "npz":
        data = np.load(filepath, allow_pickle=True)
        try:
            ids = data['sample_id']
            feats = data['joint_embedding']
            for sid, feat in zip(ids, feats):
                clean_sid = str(sid)[0:12].strip()
                feat_dict[clean_sid] = feat.reshape(-1)
        except KeyError as e:
            print(f"   [错误] {filepath} 结构异常: {e}")
            return None
            
    return feat_dict

def format_mean_std(scores):
    """辅助函数：将 0~1 的得分转换为百分比，并格式化为 'mean ± std'"""
    scores_percentage = np.array(scores) * 100
    mean_val = np.mean(scores_percentage)
    std_val = np.std(scores_percentage)
    return f"{mean_val:.2f} ± {std_val:.2f}"

# ==========================================
# 3. 数据加载与多模态独立对齐
# ==========================================
print("正在加载 CSV 标签数据...")
df_meta = pd.read_csv(CSV_PATH)
df_meta.iloc[:, 0] = df_meta.iloc[:, 0].astype(str).str[:12].str.strip()
meta_dict = dict(zip(df_meta.iloc[:, 0], df_meta.iloc[:, 1]))

print(f"准备动态解析配置中的 {len(FEATURE_PATHS)} 个特征文件...")
results = []

# ==========================================
# 4. 独立划分与随机森林训练 (5-Fold CV)
# ==========================================
for mod_name, path in FEATURE_PATHS.items():
    feat_dict = parse_feature_file(path)
    if feat_dict is None or len(feat_dict) == 0:
        continue
        
    # 单独取当前模态与标签数据的交集
    valid_ids = sorted(list(set(feat_dict.keys()) & set(meta_dict.keys())))
    
    if len(valid_ids) == 0:
        print(f"   [跳过] {mod_name} 与 CSV 无共有样本。")
        continue
        
    print(f"正在进行五折交叉验证: [{mod_name}] (共 {len(valid_ids)} 个样本)...")
    
    # 提取当前模态在有效交集下的特征和标签
    X_all = np.array([feat_dict[sid] for sid in valid_ids])
    y_all = np.array([meta_dict[sid] for sid in valid_ids])
    
    # 基于当前模态的样本量独立执行 StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(skf.split(np.zeros(len(valid_ids)), y_all))
    
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    
    # 5折训练
    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        # 初始化随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        rf.fit(X_train, y_train)
        
        # 预测
        y_pred = rf.predict(X_test)
        
        # 计算当前折指标
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        
    # 汇总计算 5 折的平均值和标准差并格式化
    results.append({
        "Modality": mod_name,
        "Sample_Count": len(valid_ids),
        "Feature_Dim": X_all.shape[1],
        "Accuracy (%)": format_mean_std(acc_list),
        "Precision (Weighted %)": format_mean_std(prec_list),
        "Recall (Weighted %)": format_mean_std(rec_list),
        "F1-Score (Weighted %)": format_mean_std(f1_list)
    })

# ==========================================
# 5. 汇总与输出总体评估表格
# ==========================================
df_results = pd.DataFrame(results)
# 按照某项指标排序一下方便查看
df_results.sort_values(by="Accuracy (%)", ascending=False, inplace=True)
df_results.set_index("Modality", inplace=True)

print("\n" + "="*95)
print(" 癌症分类预测五折交叉验证总体评估结果 (Random Forest - 独立样本对齐)")
print("="*95)
print(df_results)
print("="*95)

# 保存至本地特定的 section1 文件夹 (自动创建防止报错)
output_dir = "/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section1/"
os.makedirs(output_dir, exist_ok=True)

out_csv = os.path.join(output_dir, "S1_random_forest_multimodal_evaluation.csv")
df_results.to_csv(out_csv, encoding='utf-8-sig')
print(f"\n[成功] 聚类指标表格已保存至: {out_csv}")


########  cosine similarity  ######## 
################################################################################################

import os
import pandas as pd
import numpy as np
import torch

# ==========================================
# 1. 定义循环变量与全局存储
# ==========================================
CANCER_TYPES = [
    "BRCA", "BLCA", "UCEC", "ESCA", "CESC", "COAD", "HNSC", "KIRC", 
    "KIRP", "LGG",  "LIHC", "LUAD", "LUSC", "PAAD", "PCPG", "PRAD", 
    "SARC", "SKCM", "STAD", "THCA"
]

SURVIVAL_TYPES = ["OS", "DSS", "DFI", "PFI"]
SURVIVAL_TYPES= ["OS"]
# 用于收集所有队列计算结果的全局列表


# ==========================================
# 2. 辅助解析函数 (兼容嵌套字典和 12 位 TCGA ID)
# ==========================================
def parse_feature_file(filepath):
    fmt = "npz" if str(filepath).endswith(".npz") else "pt"
    feat_dict = {}
    
    if fmt == "pt":
        data = torch.load(filepath)
        if isinstance(data, dict):
            for sid, value in data.items():
                clean_sid = str(sid).strip()
                if isinstance(value, dict):
                    if 'input_ids' in value:
                        tensor_val = value['input_ids']
                        feat = tensor_val.cpu().detach().numpy() if isinstance(tensor_val, torch.Tensor) else np.array(tensor_val)
                        feat_dict[clean_sid] = feat.reshape(-1)
                else:
                    feat = value.cpu().detach().numpy() if isinstance(value, torch.Tensor) else np.array(value)
                    feat_dict[clean_sid] = feat.reshape(-1)
        else:
            raise ValueError(f"{filepath} 不是字典格式。")
            
    elif fmt == "npz":
        data = np.load(filepath, allow_pickle=True)
        try:
            ids = data['sample_id']
            feats = data['joint_embedding']
            for sid, feat in zip(ids, feats):
                clean_sid = str(sid).strip()
                feat_dict[clean_sid] = feat.reshape(-1)
        except KeyError:
            pass
            
    return feat_dict

# 计算余弦相似度的快速函数
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

# ==========================================
# 3. 核心双层循环逻辑
# ==========================================
all_cohort_results = []
for cancer in CANCER_TYPES:
    for surv in SURVIVAL_TYPES:
        cohort_name = f"{cancer}_{surv}"
        print(f"\n" + "="*50)
        print(f"正在处理队列: {cohort_name}")
        print("="*50)
        # --- 动态构建输入路径 ---
        CSV_PATH = f"/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/{cohort_name}_dataset.csv"
        BASE_FEAT_PATH = f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/{cancer}/{cohort_name}/align"
        FULL_EMB_PATH = f"{BASE_FEAT_PATH}/Full_modal/full_cohort/joint_embeddings.npz"
        MISSING_MODALITIES_PATHS = {
            "Missing Pathology": f"{BASE_FEAT_PATH}/Modal_miss_path/full_cohort/joint_embeddings.npz",
            "Missing Clinical": f"{BASE_FEAT_PATH}/Modal_miss_text/full_cohort/joint_embeddings.npz",
            "Missing Genomic": f"{BASE_FEAT_PATH}/Modal_miss_geno/full_cohort/joint_embeddings.npz",
            "Missing Transcriptomic": f"{BASE_FEAT_PATH}/Modal_miss_tran/full_cohort/joint_embeddings.npz",
             "Only Pathology": f"{BASE_FEAT_PATH}/Only_path/full_cohort/joint_embeddings.npz",
            "Only Clinical": f"{BASE_FEAT_PATH}/Only_text/full_cohort/joint_embeddings.npz",
            "Only Genomic": f"{BASE_FEAT_PATH}/Only_geno/full_cohort/joint_embeddings.npz",
            "Only Transcriptomic": f"{BASE_FEAT_PATH}/Only_tran/full_cohort/joint_embeddings.npz"
        }
        # --- 鲁棒性检查：确认当前组合的文件是否存在 ---
        if not os.path.exists(CSV_PATH):
            print(f"⚠️ 警告: CSV标签文件不存在，跳过该组合 -> {CSV_PATH}")
            continue
        if not os.path.exists(FULL_EMB_PATH):
            print(f"⚠️ 警告: 全模态特征文件不存在，跳过该组合 -> {FULL_EMB_PATH}")
            continue
        # --- 数据加载与交集对齐 ---
        print("正在加载 CSV 标签数据...")
        df_meta = pd.read_csv(CSV_PATH)
        df_meta.iloc[:, 0] = df_meta.iloc[:, 0].astype(str).str.strip()
        meta_dict = dict(zip(df_meta.iloc[:, 0], df_meta.iloc[:, 1]))
        common_ids = set(meta_dict.keys())
        print("正在加载全模态基准特征...")
        full_emb_dict = parse_feature_file(FULL_EMB_PATH)
        common_ids = common_ids & set(full_emb_dict.keys())
        missing_dicts = {}
        print("正在加载缺失模态特征...")
        skip_cohort = False
        for mod_name, path in MISSING_MODALITIES_PATHS.items():
            if not os.path.exists(path):
                print(f"⚠️ 警告: 缺失模态文件不存在: {path}")
                skip_cohort = True
                break
            print(f"  -> {mod_name}")
            mod_dict = parse_feature_file(path)
            missing_dicts[mod_name] = mod_dict
            common_ids = common_ids & set(mod_dict.keys())
        if skip_cohort:
            print(f"由于特征文件不全，跳过当前队列 {cohort_name} 的计算。")
            continue
        common_ids = sorted(list(common_ids))
        if len(common_ids) == 0:
            print(f"⚠️ 警告: {cohort_name} 计算出的共有样本数量为 0，跳过。")
            continue
        print(f"交集计算完成！共有 {len(common_ids)} 个样本参与 {cohort_name} 的消融分析。")
        # --- 计算余弦相似度 ---
        cancer_to_samples = {}
        for sid in common_ids:
            ctype = meta_dict[sid]
            if ctype not in cancer_to_samples:
                cancer_to_samples[ctype] = []
            cancer_to_samples[ctype].append(sid)
        for ctype, samples in cancer_to_samples.items():
            # 创建结果字典，显式包含生存类型标签
            ctype_means = {
                'Cancer Type': ctype, 
                'Survival Type': surv, 
                'Sample Count': len(samples)
            }
            for mod_name, mod_dict in missing_dicts.items():
                sims = []
                for sid in samples:
                    full_v = full_emb_dict[sid]
                    miss_v = mod_dict[sid]
                    sims.append(cosine_similarity(full_v, miss_v))
                # 计算该组下，缺失特定模态的平均相似度
                ctype_means[mod_name] = np.mean(sims)
            # 追加到全局列表，不单独在循环内落盘
            all_cohort_results.append(ctype_means)

# ==========================================
# 4. 统一合并并保存为单个文件
# ==========================================
print("\n" + "="*50)
print("正在对所有癌症及生存指标结果进行最终合并...")
print("="*50)

if all_cohort_results:
    # 转换为统一的 DataFrame
    df_final_results = pd.DataFrame(all_cohort_results)
    # 调整列顺序，使结构清晰美观
    standard_cols = [
        'Cancer Type', 'Survival Type', 'Sample Count', 
        'Missing Pathology', 'Missing Clinical', 'Missing Genomic', 'Missing Transcriptomic','Only Pathology','Only Clinical','Only Genomic','Only Transcriptomic'
    ]
    # 过滤掉由于某种特殊原因未生成的列（增强鲁棒性）
    final_cols = [col for col in standard_cols if col in df_final_results.columns]
    df_final_results = df_final_results[final_cols]
    # 定义唯一的汇总输出路径
    OUTPUT_ALL_CSV_PATH = "/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section1/S1_ablation_cosine_similarity_all.csv"
    # 自动创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(OUTPUT_ALL_CSV_PATH), exist_ok=True)
    # 保存最终合并的文件
    df_final_results.to_csv(OUTPUT_ALL_CSV_PATH, index=False)
    print(f"🎉 所有计算全部结束！全局汇总报告已成功保存至:\n👉 {OUTPUT_ALL_CSV_PATH}")
else:
    print("⚠️ 未成功收集到任何有效数据，未生成汇总文件。")

# ==========================================
# 5. 绘制可视化图表
# ==========================================
print("正在生成可视化图表...")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io


modality_cols = ['Missing Pathology', 'Missing Clinical', 'Missing Genomic', 'Missing Transcriptomic','Only Pathology','Only Clinical','Only Genomic','Only Transcriptomic']
df_loss = df_final_results.set_index('Cancer Type')[modality_cols]
df_loss = (1.0 - df_loss) * 100  

# 优化列名用于画图
df_loss.columns = ['OncoJEPA w/o Path', 'OncoJEPA w/o Path', 'OncoJEPA w/o Gen', 'OncoJEPA w/o Tran',"OncoJEPA Path-Only","OncoJEPA Clin-Only",'OncoJEPA Gen-Only','OncoJEPA Tran-Only']

# 按照最关键模态 (Transcriptomic) 的影响对癌种进行降序排序，使图表呈现对角线阶梯美感
#df_loss = df_loss.sort_values(by='Transcriptomic', ascending=False)

# 3. 绘制出版级热力图
plt.figure(figsize=(10, 8), dpi=300)
# YlOrRd 配色：颜色越红代表损失越大，模态越不可替代
sns.heatmap(df_loss, 
            cmap="YlOrRd",    
            annot=True, 
            fmt=".2f", 
            linewidths=1,
            cbar_kws={'label': 'Manifold Drift / Information Loss (%)'})

plt.title('Latent Space Deviation $\mathcal{D}$ upon Modality Dropout', 
          fontsize=14, pad=15, fontweight='bold')
plt.ylabel('TCGA Cohort', fontsize=12, fontweight='bold')
plt.xlabel('Modality_input', fontsize=12, fontweight='bold')
plt.xticks(fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

# 可以保存为高精度 PDF 以插入论文
plt.savefig("/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section1/S1_ablation_cosine_similarity_modality_importance_heatmap.pdf", format="pdf")


#########  Density 分布图
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import os
import torch

# ==========================================
# 1. 定义癌症类型列表
# ==========================================
CANCER_TYPES = [
    "BRCA", "BLCA", "UCEC", "ESCA", "CESC", "COAD", "HNSC", "KIRC", 
    "KIRP", "LGG",  "LIHC", "LUAD", "LUSC", "PAAD", "PCPG", "PRAD", 
    "SARC", "SKCM", "STAD", "THCA"
]

# ==========================================
# 2. 动态生成数据路径函数
# ==========================================
def get_feature_paths(cohort_name):
    """
    根据传入的 cohort_name 动态构建对应的数据路径字典。
    注意：Raw 模态路径现在固定指向 ALL 完整数据。
    """
    return {
        # 1. 全模态
        "OncoJEPA":      f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/{cohort_name}/{cohort_name}_OS/align/Full_modal/full_cohort/joint_embeddings.npz",
        
        # 2. 缺失模态
        "OncoJEPA w/o Gen":   f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/{cohort_name}/{cohort_name}_OS/align/Modal_miss_geno/full_cohort/joint_embeddings.npz",
        "OncoJEPA w/o Tran":  f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/{cohort_name}/{cohort_name}_OS/align/Modal_miss_tran/full_cohort/joint_embeddings.npz",
        "OncoJEPA w/o Path":  f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/{cohort_name}/{cohort_name}_OS/align/Modal_miss_path/full_cohort/joint_embeddings.npz",
        "OncoJEPA w/o Clin":  f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/{cohort_name}/{cohort_name}_OS/align/Modal_miss_text/full_cohort/joint_embeddings.npz",
        
        # 3. 单一模态
        "OncoJEPA Gen-Only":        f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/{cohort_name}/{cohort_name}_OS/align/Only_geno/full_cohort/joint_embeddings.npz",
        "OncoJEPA Tran-Only":       f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/{cohort_name}/{cohort_name}_OS/align/Only_tran/full_cohort/joint_embeddings.npz",
        "OncoJEPA Path-Only":       f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/{cohort_name}/{cohort_name}_OS/align/Only_path/full_cohort/joint_embeddings.npz",
        "OncoJEPA Clin-Only":       f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference//Full_cohort/{cohort_name}/{cohort_name}_OS/align/Only_text/full_cohort/joint_embeddings.npz", 
        
        # 4. 原始模态 (.pt 格式) —— 强制固定使用全部样本 (ALL)，后续代码会对其取交集
        "TITAN (Path)":        "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_SVS/TCGA_TITAN_first_slice_dict.pt",
        "Medgamma (Clin)":        "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/TCGA_data_clinical_patient_prompt.pt", 
        "SNN (Gen)":         "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_genome/TCGA-ALL.pt",
        "BulkFormer (Tran)":        "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_transcriptome/TCGA-ALL.pt",
        
        # 5. Baseline 晚期融合模态 (.pt 格式)
        "Concat fusion": f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/Baseline_Fusions/{cohort_name}/Concat_features.pt",
        "Mean-Pool":   f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/Baseline_Fusions/{cohort_name}/Mean_Pool_features.pt",
    }

# ==========================================
# 3. 样本交集提取与安全加载
# ==========================================
def extract_valid_ids(cohort_name):
    """
    通过读取特定癌症的 baseline 文件，提取出属于该癌症的所有病人 ID (TCGA 前 12 位)。
    """
    if cohort_name == "ALL":
        return None  # ALL 不需要取交集
    
    baseline_path = f"/home/Disk30T/chengpeng/Multimodal_JEPA/result/Baseline_Fusions/{cohort_name}/Concat_features.pt"
    if os.path.exists(baseline_path):
        try:
            data = torch.load(baseline_path, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                # 提取形如 'TCGA-XX-XXXX' 的前12位核心ID构建 Set 供快速查找
                return {str(k)[:12] for k in data.keys() if str(k).startswith("TCGA")}
        except Exception as e:
            print(f"      [警告] 提取 {cohort_name} 的 IDs 失败: {e}")
            
    print(f"      [警告] 找不到 {cohort_name} 的独立 ID 列表，Raw 数据可能无法精准过滤。")
    return None

def safe_extract_numeric(data_obj, valid_ids_set=None, depth=0):
    """
    安全提取纯数值，包含顶层字典 ID 交集过滤功能。
    """
    tensors = []
    if isinstance(data_obj, dict) or isinstance(data_obj, np.lib.npyio.NpzFile):
        for k in data_obj.keys():
            # ======= 取交集核心逻辑：只在字典第一层匹配 TCGA ID =======
            if depth == 0 and valid_ids_set is not None:
                k_str = str(k)
                if k_str.startswith("TCGA"):
                    if k_str[:12] not in valid_ids_set:
                        continue # 发现不属于该癌症的样本，直接抛弃不加载
            # ==========================================================
            
            tensors.extend(safe_extract_numeric(data_obj[k], valid_ids_set, depth + 1))
            
    elif isinstance(data_obj, torch.Tensor):
        if data_obj.is_floating_point() or data_obj.is_complex():
            tensors.append(data_obj.detach().cpu().view(-1).float().numpy())
        else:
            tensors.append(data_obj.detach().cpu().view(-1).numpy().astype(np.float32))
            
    elif isinstance(data_obj, np.ndarray) or isinstance(data_obj, list):
        try:
            tensors.append(np.array(data_obj).flatten().astype(np.float32))
        except (ValueError, TypeError):
            flat_arr = np.array(data_obj).flatten()
            numeric_list = [float(x) for x in flat_arr if _is_numeric(x)]
            if numeric_list:
                tensors.append(np.array(numeric_list, dtype=np.float32))
    return tensors

def _is_numeric(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def load_embeddings(filepath, valid_ids_set=None, is_raw=False):
    """
    加载并压平特征。如果是 Raw 数据，则传入 valid_ids_set 执行过滤。
    """
    if not os.path.exists(filepath):
        return None
    try:
        if filepath.endswith('.pt') or filepath.endswith('.pth'):
            data = torch.load(filepath, map_location='cpu', weights_only=False)
        else:
            data = np.load(filepath, allow_pickle=True)
            
        # 只有 raw 模态才需要应用 ID 过滤取交集
        ids_to_pass = valid_ids_set if is_raw else None
        numeric_tensors = safe_extract_numeric(data, ids_to_pass, depth=0)
        
        if not numeric_tensors: return None
            
        flattened = np.concatenate(numeric_tensors)
        flattened = flattened[~np.isnan(flattened) & ~np.isinf(flattened)]
        return flattened
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_all_embeddings_for_cohort(cohort_name):
    """一键加载指定队列的所有特征数据，自动处理 Raw 数据的交集"""
    paths = get_feature_paths(cohort_name)
    valid_ids = extract_valid_ids(cohort_name)
    
    loaded_embs = {}
    for name, path in paths.items():
        is_raw = any(name.startswith(prefix) for prefix in ("Raw", "TITAN", "Medgamma", "SNN", "BulkFormer"))
        emb = load_embeddings(path, valid_ids_set=valid_ids, is_raw=is_raw)
        if emb is not None and len(emb) > 0:
            loaded_embs[name] = emb
    return loaded_embs

# ==========================================
# 4. 区块绘制核心逻辑 (复用单个块绘图)
# ==========================================
def plot_cancer_block(embs_dict, cohort_name, fig, inner_gs, show_legends=True):
    """
    在一个嵌套的 inner_gs 中绘制 3 行的完整排版
    """
    if not embs_dict:
        ax_empty = fig.add_subplot(inner_gs[:, :])
        ax_empty.text(0.5, 0.5, f"No Data for {cohort_name}", ha='center', va='center', fontsize=20)
        ax_empty.axis('off')
        return
    # 为防止大图绘制过慢且保证 PDF 大小可控，降低采样率至 50000
    MAX_PTS = 50000 
    
    # === 绘制第一行 (JEPA) ===
    ax_main = fig.add_subplot(inner_gs[0, :])
    jepa_keys = [
        "OncoJEPA", "OncoJEPA w/o Gen", "OncoJEPA w/o Tran", "OncoJEPA w/o Path", "OncoJEPA w/o Clin",
        "OncoJEPA Gen-Only", "OncoJEPA Tran-Only", "OncoJEPA Path-Only", "OncoJEPA Clin-Only"
    ]
    palette = sns.color_palette("tab10", len(jepa_keys))
    
    for j, key in enumerate(jepa_keys):
        if key not in embs_dict: continue
        plot_data = embs_dict[key]
        if len(plot_data) > MAX_PTS:
            plot_data = np.random.choice(plot_data, MAX_PTS, replace=False)
            
        legend_label = f"OncoJEPA({cohort_name})" if key == "OncoJEPA(ALL)" else key
        if key == "OncoJEPA":
            sns.kdeplot(plot_data, fill=True, color='black', alpha=0.15, linewidth=3, label=legend_label, ax=ax_main)
        else:
            sns.kdeplot(plot_data, fill=False, color=palette[j], linewidth=2, alpha=0.8, label=legend_label, ax=ax_main)
            
    # 只放大字体以适配大海报版面
    ax_main.set_title(f"[{cohort_name}] OncoJEPA", fontsize=18, fontweight='bold', pad=10)
    ax_main.set_ylabel("Density", fontsize=14)
    ax_main.set_xlim([-5, 5])
    if show_legends:
        ax_main.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
    # === 绘制第二行 (Raw) ===
    raw_keys = ["TITAN (Path)", "Medgamma (Clin)", "SNN (Gen)", "BulkFormer (Tran)"]
    raw_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for i, key in enumerate(raw_keys):
        ax_sub = fig.add_subplot(inner_gs[1, i])
        if key not in embs_dict:
            ax_sub.set_title(key, fontsize=12, linestyle='--')
            continue
            
        plot_data = embs_dict[key]
        if len(plot_data) > MAX_PTS:
            plot_data = np.random.choice(plot_data, MAX_PTS, replace=False)
            
        sns.kdeplot(plot_data, fill=True, color=raw_colors[i], alpha=0.5, linewidth=2, ax=ax_sub)
        ax_sub.set_title(key, fontsize=12, fontweight='bold')
        if "Gen" in key or "Tran" in key:
            ax_sub.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            
    # === 绘制第三行 (Baseline) ===
    ax_baseline = fig.add_subplot(inner_gs[2, :])
    baseline_keys = ["Concat fusion", "Mean-Pool"]
    baseline_colors = ["#9467bd", "#e377c2"]
    
    plotted_baseline = False
    for j, key in enumerate(baseline_keys):
        if key not in embs_dict: continue
        plot_data = embs_dict[key]
        if len(plot_data) > MAX_PTS:
            plot_data = np.random.choice(plot_data, MAX_PTS, replace=False)
            
        # Z-Score Standardize
        plot_data = (plot_data - np.mean(plot_data)) / (np.std(plot_data) + 1e-8)
        sns.kdeplot(plot_data, fill=True, color=baseline_colors[j], linewidth=2, alpha=0.4, label=key, ax=ax_baseline)
        plotted_baseline = True
        
    if plotted_baseline:
        ax_baseline.set_title(f"[{cohort_name}] Baselines (Z-Score)", fontsize=18, fontweight='bold', pad=10)
        ax_baseline.set_ylabel("Density", fontsize=14)
        ax_baseline.set_xlim([-5, 5])
        if show_legends:
            ax_baseline.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)

# ==========================================
# 5. 主运行流程
# ==========================================
if __name__ == "__main__":
    # 定义输出目录
    output_dir = "/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section1"
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ------------------------------------------
    # 步骤 A: 绘制 "ALL" (独立的主图)
    # ------------------------------------------
    print("\n>>> [步骤 A] 正在处理主队列 ALL ...")
    all_embs = load_all_embeddings_for_cohort("ALL")
    if all_embs:
        fig_all = plt.figure(figsize=(20, 16))
        gs_all = gridspec.GridSpec(3, 4, height_ratios=[1.2, 1, 1.2], hspace=0.4)
        plot_cancer_block(all_embs, "ALL", fig_all, gs_all, show_legends=True)
        main_figure_path = os.path.join(output_dir, "Fig_Main_ALL_Density_Distribution.pdf")
        plt.savefig(main_figure_path, dpi=300, bbox_inches='tight')
        plt.close(fig_all)
        print(f"✅ ALL 图表保存至: {main_figure_path}")
    
    # ------------------------------------------
    # 步骤 B: 绘制 20 个癌症的 5x4 补充大图
    # ------------------------------------------
    print("\n>>> [步骤 B] 正在生成 20 种癌症的补充网格大图 (5行4列) ...")

    # 构建巨型海报尺寸画布 (宽度约 48 英寸，高度约 50 英寸)，以保障 20 个块的清晰度
    fig_grid = plt.figure(figsize=(48, 55))

    # 外层网格：5 行 4 列
    outer_gs = gridspec.GridSpec(5, 4, wspace=0.35, hspace=0.35)

    for idx, cancer in enumerate(CANCER_TYPES):
        print(f"    [{idx+1}/{len(CANCER_TYPES)}] 正在处理并拼接区块: {cancer} ...")
        
        # 加载时已经自动利用 Baseline 的 IDs 过滤了 Raw
        cancer_embs = load_all_embeddings_for_cohort(cancer)
        
        # 为当前癌症在整体 5x4 网格中建立内部的 3 行结构
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            3, 4, 
            subplot_spec=outer_gs[idx], 
            height_ratios=[1.2, 1, 1.2], 
            hspace=0.45
        )
        
        # 只在第一列的图展示图例，防止排版过于拥挤杂乱
        show_legends_flag = (idx % 4 == 3) 
        plot_cancer_block(cancer_embs, cancer, fig_grid, inner_gs, show_legends=show_legends_flag)
        
    grid_pdf_path = os.path.join(output_dir, "Fig_Main_ALL_Density_Distribution_Supplementary_.pdf")
    print("\n>>> ⏳ 正在渲染并保存巨型 PDF (可能需要几分钟) ...")

    # 带有 bbox_inches='tight' 可确保边缘白边被裁剪
    fig_grid.savefig(grid_pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig_grid)

    print("\n==============================================")
    print("🎉 所有任务执行完毕！")
    print(f"📊 主图 (ALL): {main_figure_path}")
    print(f"📕 20病种网格大图: {grid_pdf_path}")
    print("==============================================")






