library(yaml)

cancer_type <- c("BRCA","BLCA","UCEC","ESCA","CESC","COAD","GBM","HNSC",
                 "KIRC","KIRP","LGG","LIHC","LUAD","LUSC","OV","PAAD",
                 "PCPG","PRAD","READ","SARC","SKCM","STAD","TGCT","THCA","THYM")
delete_cancer_type <-c("OV", "GBM" , "THYM" , "TGCT" , "READ" )
 cancer_type <-setdiff(cancer_type,delete_cancer_type)

for(i in cancer_type){ 
  tt <-read.csv(paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/",i,"_OS_dataset.csv"))  
  print(paste0(i, " Total_num: ",nrow(tt)," Death_num: ",nrow(subset(tt,survival_event==1)))) 
}



# batch_list <- list(
#   # ========== 【阵营 A：小 Batch 噪声红利，保留低学习率】 ==========
#   # 适用：小样本或异质性极强的困难户
#   ESCA = list(align_batch_size_count = 24, prognosis_batch_size_count = 32,  lr_set=0.00003, align_epoch_num=40, prognosis_epoch_num=80),
#   CESC = list(align_batch_size_count = 32, prognosis_batch_size_count = 200,  lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   HNSC = list(align_batch_size_count = 64, prognosis_batch_size_count = 64,  lr_set=0.00008, align_epoch_num=40, prognosis_epoch_num=80),
#   LGG  = list(align_batch_size_count = 64, prognosis_batch_size_count = 64,  lr_set=0.00008, align_epoch_num=40, prognosis_epoch_num=80),
#   BRCA = list(align_batch_size_count = 90, prognosis_batch_size_count = 128, lr_set=0.00008, align_epoch_num=40, prognosis_epoch_num=80),
#   LUSC = list(align_batch_size_count = 64, prognosis_batch_size_count = 128, lr_set=0.00008, align_epoch_num=40, prognosis_epoch_num=80),
#   
#   # ========== 【阵营 B：大 Batch 全局视野，必须恢复 0.0005 高学习率】 ==========
#   # 适用：事件数充足、大样本，需要快速、平滑收敛的癌种（对标第三列的高分状态）
#   KIRC = list(align_batch_size_count = 64, prognosis_batch_size_count = 200, lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   LIHC = list(align_batch_size_count = 32, prognosis_batch_size_count = 256, lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   SARC = list(align_batch_size_count = 32, prognosis_batch_size_count = 256, lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   SKCM = list(align_batch_size_count = 32, prognosis_batch_size_count = 256, lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   STAD = list(align_batch_size_count = 32, prognosis_batch_size_count = 256, lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   LUAD = list(align_batch_size_count = 64, prognosis_batch_size_count = 200, lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   COAD = list(align_batch_size_count = 64, prognosis_batch_size_count = 200, lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   BLCA = list(align_batch_size_count = 64, prognosis_batch_size_count = 200, lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   UCEC = list(align_batch_size_count = 64, prognosis_batch_size_count = 200, lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   KIRP = list(align_batch_size_count = 32, prognosis_batch_size_count = 256, lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   PAAD = list(align_batch_size_count = 24, prognosis_batch_size_count = 150, lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   
#   # ========== 【阵营 C：极低死亡率特例，特殊关照】 ==========
#   # THCA 在第三列用大 LR 跑出了 0.851 的高分，必须用大 LR
#   THCA = list(align_batch_size_count = 64, prognosis_batch_size_count = 256, lr_set=0.0005, align_epoch_num=40, prognosis_epoch_num=80),
#   PCPG = list(align_batch_size_count = 24, prognosis_batch_size_count = 256, lr_set=0.00008, align_epoch_num=40, prognosis_epoch_num=80),
#   # PRAD 在第一列用低 LR (0.00008) 跑出了 0.736，比高 LR 表现好，予以保留低 LR
#   PRAD = list(align_batch_size_count = 64, prognosis_batch_size_count = 256, lr_set=0.00008, align_epoch_num=40, prognosis_epoch_num=80),
#   ALL = list(align_batch_size_count = 90, prognosis_batch_size_count = 128, lr_set=0.00008, align_epoch_num=40, prognosis_epoch_num=80)
# )
# 
# batch_list <- list(
#   
#   # ========== 【阵营 A：极低死亡率特例 —— 必须强制 Full-Batch 全批量】 ==========
#   # 逻辑：训练集极小且事件极少。Prognosis Batch 设定必须大于等于训练集大小，确保前向传播包含所有死亡样本。
#   PCPG = list(align_batch_size_count = 32, prognosis_batch_size_count = 200, lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80), # Train约149，设200覆盖
#   PRAD = list(align_batch_size_count = 64, prognosis_batch_size_count = 500, lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80), # Train约452，设500覆盖
#   THCA = list(align_batch_size_count = 64, prognosis_batch_size_count = 500, lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80), # Train约464，设500覆盖
#   
#   # ========== 【阵营 B：微型队列 —— 恢复 Epoch 更新步数，禁止一轮游】 ==========
#   # 逻辑：总样本不足 300，坚决杜绝 150/200/256 的大 Batch，否则每 Epoch 仅更新 1 次，无法寻找最优解。
#   ESCA = list(align_batch_size_count = 32, prognosis_batch_size_count = 32, lr_set=0.00003,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80),
#   PAAD = list(align_batch_size_count = 32, prognosis_batch_size_count = 32, lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
#   SARC = list(align_batch_size_count = 32, prognosis_batch_size_count = 64, lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80), # Train 216，用64保障更新步数
#   
#   # ========== 【阵营 C：中型稳健队列 —— 统一降阶至 Batch 64】 ==========
#   # 逻辑：总样本 300~600 之间，配合你的高学习率 (0.0005)，Batch 64 能榨取稳定的多组学特征噪音，避免见顶阴跌。
#   CESC = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
#   KIRP = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
#   BLCA = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
#   LIHC = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
#   SKCM = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
#   STAD = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
#   COAD = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
#   UCEC = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
#   LGG  = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80),
#   
#   # ========== 【阵营 D：大型充足队列 —— 安全激活大批量与全局流形视野】 ==========
#   # 逻辑：总样本 > 600，模型有底气切分出超过 90~128 样本的 Batch，且依然有丰富的死亡事件计算 Cox Loss。
#   HNSC = list(align_batch_size_count = 64, prognosis_batch_size_count = 90,  lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80),
#   LUAD = list(align_batch_size_count = 64, prognosis_batch_size_count = 90,  lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
#   LUSC = list(align_batch_size_count = 64, prognosis_batch_size_count = 90,  lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80),
#   KIRC = list(align_batch_size_count = 90, prognosis_batch_size_count = 128, lr_set=0.0005,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
#   BRCA = list(align_batch_size_count = 90, prognosis_batch_size_count = 128, lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80),
#   
#   ALL  = list(align_batch_size_count = 90, prognosis_batch_size_count = 90,  lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80)
# )

# 
batch_list <- list(

  # ========== 【阵营 A：极低死亡率特例 —— 必须强制 Full-Batch 全批量】 ==========
  # 逻辑：训练集极小且事件极少。Prognosis Batch 设定必须大于等于训练集大小，确保前向传播包含所有死亡样本。
  PCPG = list(align_batch_size_count = 32, prognosis_batch_size_count = 200, lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80), # Train约149，设200覆盖
  PRAD = list(align_batch_size_count = 64, prognosis_batch_size_count = 500, lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80), # Train约452，设500覆盖
  THCA = list(align_batch_size_count = 64, prognosis_batch_size_count = 500, lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80), # Train约464，设500覆盖

  # ========== 【阵营 B：微型队列 —— 恢复 Epoch 更新步数，禁止一轮游】 ==========
  # 逻辑：总样本不足 300，坚决杜绝 150/200/256 的大 Batch，否则每 Epoch 仅更新 1 次，无法寻找最优解。
  ESCA = list(align_batch_size_count = 32, prognosis_batch_size_count = 32, lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80),
  PAAD = list(align_batch_size_count = 32, prognosis_batch_size_count = 32, lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
  SARC = list(align_batch_size_count = 32, prognosis_batch_size_count = 64, lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80), # Train 216，用64保障更新步数

  # ========== 【阵营 C：中型稳健队列 —— 统一降阶至 Batch 64】 ==========
  # 逻辑：总样本 300~600 之间，配合你的高学习率 (0.0005)，Batch 64 能榨取稳定的多组学特征噪音，避免见顶阴跌。
  CESC = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
  KIRP = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
  BLCA = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
  LIHC = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
  SKCM = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
  STAD = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
  COAD = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
  UCEC = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
  LGG  = list(align_batch_size_count = 64, prognosis_batch_size_count = 64, lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80),

  # ========== 【阵营 D：大型充足队列 —— 安全激活大批量与全局流形视野】 ==========
  # 逻辑：总样本 > 600，模型有底气切分出超过 90~128 样本的 Batch，且依然有丰富的死亡事件计算 Cox Loss。
  HNSC = list(align_batch_size_count = 64, prognosis_batch_size_count = 90,  lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80),
  LUAD = list(align_batch_size_count = 64, prognosis_batch_size_count = 90,  lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
  LUSC = list(align_batch_size_count = 64, prognosis_batch_size_count = 90,  lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80),
  KIRC = list(align_batch_size_count = 90, prognosis_batch_size_count = 128, lr_set=0.00008,wd_set=0.05,  align_epoch_num=40, prognosis_epoch_num=80),
  BRCA = list(align_batch_size_count = 90, prognosis_batch_size_count = 128, lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80),

  ALL  = list(align_batch_size_count = 90, prognosis_batch_size_count = 90,  lr_set=0.00008,wd_set=0.05, align_epoch_num=40, prognosis_epoch_num=80)
)

########################   1. fold level train shell
train_sh_file <- "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/1.all_cancer_train_by_fold.sh"
csv_sh_file <- "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/train_sample_csv_create.sh"
template_file <- "/home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/configs/demo/train_BRCA_v3_5.yaml"

# =========================================================================
# 💡 修改点 1: 初始化 Bash 脚本，加入 xargs 并发控制框架
# =========================================================================
write("#!/bin/bash", train_sh_file, append = FALSE)
write("# 使用说明: bash 1.all_cancer_train_by_fold.sh [并发数量]", train_sh_file, append = TRUE)
write("# 如果不传参数，默认同时并行运行 4 组", train_sh_file, append = TRUE)
write("N_PARALLEL=${1:-1}", train_sh_file, append = TRUE)
write("echo \"🚀 开始执行训练任务，当前最大并行任务数: $N_PARALLEL\"", train_sh_file, append = TRUE)
write("echo \"---------------------------------------------------\"", train_sh_file, append = TRUE)
# 开始写入 heredoc，交由 xargs 调度
write("cat << 'EOF' | xargs -I {} -P $N_PARALLEL bash -c \"{}\"", train_sh_file, append = TRUE)

write("## iter run", csv_sh_file, append = FALSE)

for(time_type in c("OS","DSS","DFI","PFI")) {              
  for(i in c(cancer_type,"ALL")){
    
    cat("\n=== 处理: ", i, " - ", time_type, " ===\n", sep = "")
    
    yaml_file = paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/config/split_by_fold/train_", i, "_", time_type, ".yaml")
    
    # 1. 检查模板文件
    if (!file.exists(template_file) || file.info(template_file)$size == 0) {
      stop("模板文件为空或不存在: ", template_file)
    }
    
    # 2. 获取基因数量
    gene_file <- paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_genome/TCGA-", i, "_gene_list.txt")
    if (!file.exists(gene_file)) {
      cat("⚠️  跳过 ", i, "，基因文件不存在\n", sep = "")
      next
    }
    
    line_count <- as.integer(system(paste("wc -l <", shQuote(gene_file)), intern = TRUE))
    cat("基因数量: ", line_count, "\n", sep = "")
    
    # 3. 使用临时文件处理
    temp_file <- tempfile(fileext = ".yaml")
    file.copy(template_file, temp_file, overwrite = TRUE)
    
    
    
    # 读取 YAML
    cfg <- read_yaml(temp_file)
    
    # ===== 一一对应你原来的 sed =====
    
    # BRCA -> i
    cfg$output_path <- gsub("Model_train/BRCA/", paste0("result_all_emb/Model_train/",i,"/"), cfg$output_path)
    
    # _OS_ -> _time_type_
    cfg$data$csv_path <- gsub("_OS_", paste0("_", time_type, "_"), cfg$data$csv_path)
    
    # geno_raw_dim
    cfg$data$csv_path <- gsub("BRCA", i, cfg$data$csv_path)
    cfg$models$geno_raw_dim <- as.integer(line_count + 1)
    cfg$data$geno_pt_path <- gsub("BRCA", i, cfg$data$geno_pt_path)
    cfg$data$tran_pt_path <- gsub("BRCA", i, cfg$data$tran_pt_path)
    # align_batch_size
    cfg$data$align_batch_size <-
      as.integer(as.numeric(batch_list[[i]]["align_batch_size_count"]))
    
    # prognosis lr（✅ 精确、无副作用）
    cfg$optimization$prognosis$lr <-
      as.numeric(batch_list[[i]]["lr_set"])
    cfg$optimization$t2i_weight_decay <-
      as.numeric(batch_list[[i]]["wd_set"])
    cfg$optimization$t2i_final_weight_decay <-
      as.numeric(batch_list[[i]]["wd_set"])
    
    # prognosis_batch_size
    cfg$data$prognosis_batch_size <-
      as.integer(as.numeric(batch_list[[i]]["prognosis_batch_size_count"]))
    
    
    
    old_scipen <- options("scipen")$scipen
    options(scipen = 999)
    # ===== 写回 =====
    write_yaml(cfg, temp_file)
    options(scipen = old_scipen)
    # 4. 移动临时文件到目标位置
    file.copy(temp_file, yaml_file, overwrite = TRUE)
    file.remove(temp_file)
    
    # 5. 验证结果
    if (file.exists(yaml_file) && file.info(yaml_file)$size > 0) {
      cat("✅ 配置文件生成成功: ", basename(yaml_file), "\n")
    } else {
      cat("❌ 配置文件生成失败: ", basename(yaml_file), "\n")
      next
    }
    
    # 6. 写入训练脚本
   # if(time_type=="OS"){
      align_cmd <- sprintf("/home/Disk14T/chengpeng/miniconda3/envs/M3_JEPA/bin/python /home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/main.py --fname %s --task align --run_name %s_%s",
                           yaml_file, i, time_type)
      prognosis_cmd <- sprintf("/home/Disk14T/chengpeng/miniconda3/envs/M3_JEPA/bin/python /home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/main.py --fname %s --task prognosis --run_name %s_%s",
                               yaml_file, i, time_type)
      
      # =========================================================================
      # 💡 修改点 2: 将 align 和 prognosis 合并为一个串行任务组 (使用 && 符号)
      # =========================================================================
      group_cmd <- sprintf("%s && %s", align_cmd, prognosis_cmd)
      
      write(group_cmd, train_sh_file, append = TRUE)
  # }
    
    # 7. 写入样本创建脚本
    r_cmd <- sprintf("/home/Disk14T/chengpeng/miniconda3/envs/general_use/bin/Rscript /home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/train_sample_csv_create.R --input /home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/Survival_SupplementalTable_S1_20171025_xena_sp --output /home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/%s_%s_dataset.csv --cancer_type %s --survival_type %s",
                     i, time_type, i, time_type)
    
    write(r_cmd, csv_sh_file, append = TRUE)
    
  }
 # r_cmd <- sprintf("/home/Disk14T/chengpeng/miniconda3/envs/general_use/bin/Rscript /home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/train_sample_csv_create.R --input /home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/Survival_SupplementalTable_S1_20171025_xena_sp --output /home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/%s_%s_dataset.csv --cancer_type %s --survival_type %s",
 #                  "TCGA_ALL", time_type, "all", time_type)
  
  write(r_cmd, csv_sh_file, append = TRUE)
}

# =========================================================================
# 💡 修改点 3: 闭合 heredoc 块，完成脚本编写
# =========================================================================
write("EOF", train_sh_file, append = TRUE)
write("echo \"🎉 所有模型训练任务执行完毕!\"", train_sh_file, append = TRUE)

cat("\n=== 处理完成 ===\n")



########################   1.2. fold level inference
library(yaml)

input_path = "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_train/"
output_path ="/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_inference/Split_by_fold/"
yaml_path <-"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_inference/configs/Split_by_fold/"

inference_sh_file <- paste0(output_path,"/../../1.2.all_cancer_inference_by_fold.sh")
write("#!/bin/bash", inference_sh_file, append = FALSE)


for(each_cancer_type in list.files(input_path)){
  
  gene_file <- paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_genome/TCGA-", each_cancer_type, "_gene_list.txt")
  
  
  line_count <- as.integer(system(paste("wc -l <", shQuote(gene_file)), intern = TRUE))
  
  
  cancer_time <- list.files(paste0(input_path,"/",each_cancer_type))
  
  for(each_cancer_time in cancer_time){
    
    for(fold in paste0("fold_",0:4)){
      
      val_csv <-paste0(input_path,"/",each_cancer_type,"/",each_cancer_time,"/",fold,"/val.csv")
      align_pth <-paste0(input_path,"/",each_cancer_type,"/",each_cancer_time,"/",fold,"/align-best.pth.tar")
      prognosis_pth <-paste0(input_path,"/",each_cancer_type,"/",each_cancer_time,"/",fold,"/prognosis-best.pth.tar")
      
      for(each_test_mode in c("prognosis","align")){
        
      
          for( avi_modal_name  in c("Full_modal","Modal_miss_path","Modal_miss_text","Modal_miss_geno","Modal_miss_tran","Only_path","Only_text","Only_geno","Only_tran")){
            csv_path = val_csv
            path_pt_path = "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_SVS/TCGA_TITAN_first_slice_dict.pt"
            text_pt_path = "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/TCGA_data_clinical_patient_prompt.pt"
            geno_pt_path = paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_genome/TCGA-",each_cancer_type,".pt")
            tran_pt_path = paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_transcriptome/TCGA-",each_cancer_type,".pt")
            
            if(avi_modal_name=="Modal_miss_path"){
              path_pt_path = NULL
            }else if(avi_modal_name=="Modal_miss_text"){
              text_pt_path = NULL
            }else if(avi_modal_name=="Modal_miss_geno"){
              geno_pt_path = NULL
            }else if(avi_modal_name=="Modal_miss_tran"){
              tran_pt_path = NULL
            }else if(avi_modal_name=="Only_path"){
              tran_pt_path = NULL
              text_pt_path = NULL
              geno_pt_path = NULL
            }else if(avi_modal_name=="Only_text"){
              tran_pt_path = NULL
              path_pt_path = NULL
              geno_pt_path = NULL
            }else if(avi_modal_name=="Only_geno"){
              tran_pt_path = NULL
              text_pt_path = NULL
              path_pt_path = NULL
            }else if(avi_modal_name=="Only_tran"){
              path_pt_path = NULL
              text_pt_path = NULL
              geno_pt_path = NULL
            } else{
              NULL
            }

          run_name = fold
          each_output_path=paste0(output_path,"/",each_cancer_type,"/",each_cancer_time,"/",each_test_mode,"/",avi_modal_name,"/")
          each_model_path=paste0(input_path,"/",each_cancer_type,"/",each_cancer_time,"/",fold,"/",each_test_mode,"-best.pth.tar")
          each_yaml_path <-paste0(yaml_path,"/",each_cancer_type,"_",each_cancer_time,"_",each_test_mode,"_",avi_modal_name,"_",fold,".yaml") 
          
          yaml_data <- list(
            task_mode = each_test_mode,
            run_name = run_name,
            model_path = each_model_path,
            output_path = each_output_path,
            use_bfloat16 = TRUE,
            
            data = list(
              csv_path = csv_path,
              batch_size = 64L, # 使用 L 强制声明为整数，以保持格式整洁
              num_workers = 4L,
              pin_mem = TRUE,
              path_pt_path = path_pt_path,
              text_pt_path = text_pt_path,
              geno_pt_path = geno_pt_path,
              tran_pt_path = tran_pt_path
            ),
            
            models = list(
              text_encoder_name = "/home/Disk30T/chengpeng/Multimodal_JEPA/medgemma-1.5-4b-it",
              path_dim = 768L,
              geno_raw_dim = as.integer(line_count + 1),
              geno_dim = 512L,
              tran_dim = 643L
            ),
            
            MoE = list(
              num_experts = 8L,
              k = 2L,
              drop = 0.1,
              hidden_size = 1024L
            )
          )
          write_yaml(yaml_data, file = each_yaml_path)
          each_run <-paste0("/home/Disk14T/chengpeng/miniconda3/envs/M3_JEPA/bin/python /home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/src/inference_v5_3.py --fname " ,each_yaml_path)
          write(each_run, inference_sh_file, append = T)
          if(each_test_mode=="prognosis"){
            each_run <-paste0("/home/Disk14T/chengpeng/miniconda3/envs/M3_JEPA/bin/python /home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/src/evaluate_cindex.py --scores ", each_output_path,"/",run_name,"/prognosis_risk_scores.csv "," --yaml " ,each_yaml_path)
            write(each_run, inference_sh_file, append = T)
          }
        }
        
      }
    }
  }
}


########################   1.2.2 fold level inference train risk score
library(yaml)

input_path = "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_train/"
output_path ="/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_inference/Split_by_fold/"
yaml_path <-"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_inference/configs/Split_by_fold/train_config"

inference_sh_file <- paste0(output_path,"/../../1.3.all_cancer_inference_by_fold_train_riskscore.sh")
write("#!/bin/bash", inference_sh_file, append = FALSE)


for(each_cancer_type in list.files(input_path)){
  
  gene_file <- paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_genome/TCGA-", each_cancer_type, "_gene_list.txt")
  
  
  line_count <- as.integer(system(paste("wc -l <", shQuote(gene_file)), intern = TRUE))
  
  
  cancer_time <- list.files(paste0(input_path,"/",each_cancer_type))
  
  for(each_cancer_time in cancer_time){
    
    for(fold in paste0("fold_",0:4)){
      
      val_csv <-paste0(input_path,"/",each_cancer_type,"/",each_cancer_time,"/",fold,"/train.csv")
      align_pth <-paste0(input_path,"/",each_cancer_type,"/",each_cancer_time,"/",fold,"/align-best.pth.tar")
      prognosis_pth <-paste0(input_path,"/",each_cancer_type,"/",each_cancer_time,"/",fold,"/prognosis-best.pth.tar")
      
      for(each_test_mode in c("prognosis")){
        
        
        for( avi_modal_name  in c("Full_modal","Modal_miss_path","Modal_miss_text","Modal_miss_geno","Modal_miss_tran","Only_path","Only_text","Only_geno","Only_tran")){
          csv_path = val_csv
          path_pt_path = "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_SVS/TCGA_TITAN_first_slice_dict.pt"
          text_pt_path = "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/TCGA_data_clinical_patient_prompt.pt"
          geno_pt_path = paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_genome/TCGA-",each_cancer_type,".pt")
          tran_pt_path = paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_transcriptome/TCGA-",each_cancer_type,".pt")
          
          if(avi_modal_name=="Modal_miss_path"){
            path_pt_path = NULL
          }else if(avi_modal_name=="Modal_miss_text"){
            text_pt_path = NULL
          }else if(avi_modal_name=="Modal_miss_geno"){
            geno_pt_path = NULL
          }else if(avi_modal_name=="Modal_miss_tran"){
            tran_pt_path = NULL
          }else if(avi_modal_name=="Only_path"){
            tran_pt_path = NULL
            text_pt_path = NULL
            geno_pt_path = NULL
          }else if(avi_modal_name=="Only_text"){
            tran_pt_path = NULL
            path_pt_path = NULL
            geno_pt_path = NULL
          }else if(avi_modal_name=="Only_geno"){
            tran_pt_path = NULL
            text_pt_path = NULL
            path_pt_path = NULL
          }else if(avi_modal_name=="Only_tran"){
            path_pt_path = NULL
            text_pt_path = NULL
            geno_pt_path = NULL
          } else{
            NULL
          }
          
          run_name = fold
          each_output_path=paste0(output_path,"/",each_cancer_type,"/",each_cancer_time,"/",each_test_mode,"/",avi_modal_name,"_train_riskscore/")
          each_model_path=paste0(input_path,"/",each_cancer_type,"/",each_cancer_time,"/",fold,"/",each_test_mode,"-best.pth.tar")
          each_yaml_path <-paste0(yaml_path,"/",each_cancer_type,"_",each_cancer_time,"_",each_test_mode,"_",avi_modal_name,"_",fold,".yaml") 
          
          yaml_data <- list(
            task_mode = each_test_mode,
            run_name = run_name,
            model_path = each_model_path,
            output_path = each_output_path,
            use_bfloat16 = TRUE,
            
            data = list(
              csv_path = csv_path,
              batch_size = 64L, # 使用 L 强制声明为整数，以保持格式整洁
              num_workers = 4L,
              pin_mem = TRUE,
              path_pt_path = path_pt_path,
              text_pt_path = text_pt_path,
              geno_pt_path = geno_pt_path,
              tran_pt_path = tran_pt_path
            ),
            
            models = list(
              text_encoder_name = "/home/Disk30T/chengpeng/Multimodal_JEPA/medgemma-1.5-4b-it",
              path_dim = 768L,
              geno_raw_dim = as.integer(line_count + 1),
              geno_dim = 512L,
              tran_dim = 643L
            ),
            
            MoE = list(
              num_experts = 8L,
              k = 2L,
              drop = 0.1,
              hidden_size = 1024L
            )
          )
          write_yaml(yaml_data, file = each_yaml_path)
          each_run <-paste0("/home/Disk14T/chengpeng/miniconda3/envs/M3_JEPA/bin/python /home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/src/inference_v5_3.py --fname " ,each_yaml_path)
          write(each_run, inference_sh_file, append = T)
          if(each_test_mode=="prognosis"){
            each_run <-paste0("/home/Disk14T/chengpeng/miniconda3/envs/M3_JEPA/bin/python /home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/src/evaluate_cindex.py --scores ", each_output_path,"/",run_name,"/prognosis_risk_scores.csv "," --yaml " ,each_yaml_path)
            write(each_run, inference_sh_file, append = T)
          }
        }
        
      }
    }
  }
}



###########   full cohort part 

####### full_cohort : find best epoch
# 加载所需的包
library(tidyverse)


base_dir <- "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_train/"

# ==========================================
# 1. 使用正则表达式同时匹配两种文件
# ==========================================
csv_files <- list.files(
  path = base_dir, 
  pattern = "(align|prognosis)_cross_validation_results\\.csv$", 
  recursive = TRUE, 
  full.names = TRUE
)

# ==========================================
# 2. 定义处理单个文件的函数 (核心修改区域)
# ==========================================
process_epoch_data <- function(file_path) {
  
  # 提取文件名和任务类型
  file_name <- basename(file_path)
  task_type <- ifelse(str_detect(file_name, "align"), "Align", "Prognosis")
  
  # 解析目录获取癌症和预后类型
  sub_dir <- basename(dirname(file_path))             
  parent_dir <- basename(dirname(dirname(file_path))) 
  prog_type <- str_replace(sub_dir, paste0("^", parent_dir, "_"), "")
  
  # 读取 CSV 数据
  data <- read_csv(file_path, show_col_types = FALSE)
  names(data)[1:2] <- c("Fold", "Best_Epoch")
  
  # 【关键修改】：不再读取 Average 行，而是提取真正的 Fold_0 到 Fold_4 进行计算
  fold_data <- data %>%
    filter(str_detect(Fold, "^Fold_")) %>%  # 过滤掉 Average 和 Std_Dev 行
    mutate(Best_Epoch = as.numeric(Best_Epoch))
  
  # 计算中位数 (Median)
  median_epoch <- median(fold_data$Best_Epoch, na.rm = TRUE)
  
  # 计算全量模型建议使用的 Epoch (乘以 0.8 并四舍五入)
  adjusted_epoch <- round(median_epoch )
  
  # 将结果打包为单行表格返回
  result_data <- tibble(
    Cancer = parent_dir,
    Prognosis = prog_type,
    Task = task_type,
    Median_Epoch = median_epoch,
    Full_Train_Epoch = adjusted_epoch
  )
  
  return(result_data)
}

# ==========================================
# 3. 批量处理并合并数据
# ==========================================
raw_summary_table <- map_df(csv_files, process_epoch_data)

# ==========================================
# 4. 数据透视与整理
# ==========================================
# 将长表转为宽表，这里同时把 Median 和计算好的 Full_Train 展开
final_table <- raw_summary_table %>%
  pivot_wider(
    names_from = Task, 
    values_from = c(Median_Epoch, Full_Train_Epoch),
    names_glue = "{Task}_{.value}" # 自定义生成的列名格式，例如 Align_Median_Epoch
  ) %>%
  # 按癌症类型和预后排序
  arrange(Cancer, Prognosis)

# ==========================================
# 5. 打印结果
# ==========================================
print(head(final_table))

# ==========================================
# 5. 导出最终统计表
# ==========================================

batch_full_list <-batch_list
for(i in 1:nrow(final_table)){
  
  batch_full_list[[final_table$Cancer[i]]][[final_table$Prognosis[i]]][["align_epoch_num"]]<-max(10,ceiling(final_table$Align_Median_Epoch[i]))
  batch_full_list[[final_table$Cancer[i]]][[final_table$Prognosis[i]]][["prognosis_epoch_num"]] <-max(10,ceiling(final_table$Prognosis_Median_Epoch[i]))
}


########

########  full_cohort  level train shell


cancer_type <- c("BRCA","BLCA","UCEC","ESCA","CESC","COAD","GBM","HNSC",
                 "KIRC","KIRP","LGG","LIHC","LUAD","LUSC","OV","PAAD",
                 "PCPG","PRAD","READ","SARC","SKCM","STAD","TGCT","THCA","THYM")
delete_cancer_type <-c("OV", "GBM" , "THYM" , "TGCT" , "READ" )
cancer_type <-setdiff(cancer_type,delete_cancer_type)


train_sh_file <- "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/2.all_cancer_train_full_cohort.sh"
template_file <- "/home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/configs/demo/train_BRCA_v3_5.yaml"

# =========================================================================
# 💡 修改点 1: 初始化 Bash 脚本，加入 xargs 并发控制框架
# =========================================================================
write("#!/bin/bash", train_sh_file, append = FALSE)
write("# 使用说明: bash 2.all_cancer_train_full_cohort.sh [并发数量]", train_sh_file, append = TRUE)
write("# 如果不传参数，默认同时并行运行 4 组", train_sh_file, append = TRUE)
write("N_PARALLEL=${1:-1}", train_sh_file, append = TRUE)
write("echo \"🚀 开始执行训练任务，当前最大并行任务数: $N_PARALLEL\"", train_sh_file, append = TRUE)
write("echo \"---------------------------------------------------\"", train_sh_file, append = TRUE)
# 开始写入 heredoc，交由 xargs 调度
write("cat << 'EOF' | xargs -I {} -P $N_PARALLEL bash -c \"{}\"", train_sh_file, append = TRUE)


for(time_type in c("OS","DSS","DFI","PFI")) {              
  for(i in c(cancer_type,"ALL")){
    
    cat("\n=== 处理: ", i, " - ", time_type, " ===\n", sep = "")
    
    yaml_file = paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/config/full_cohort/train_", i, "_full_cohort_", time_type, ".yaml")
    
    # 1. 检查模板文件
    if (!file.exists(template_file) || file.info(template_file)$size == 0) {
      stop("模板文件为空或不存在: ", template_file)
    }
    
    # 2. 获取基因数量
    gene_file <- paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_genome/TCGA-", i, "_gene_list.txt")
    if (!file.exists(gene_file)) {
      cat("⚠️  跳过 ", i, "，基因文件不存在\n", sep = "")
      next
    }
    
    line_count <- as.integer(system(paste("wc -l <", shQuote(gene_file)), intern = TRUE))
    cat("基因数量: ", line_count, "\n", sep = "")
    
    # 3. 使用临时文件处理
    temp_file <- tempfile(fileext = ".yaml")
    file.copy(template_file, temp_file, overwrite = TRUE)
    
    # 读取 YAML
    cfg <- read_yaml(temp_file)
    
    # ===== 一一对应你原来的 sed =====
    
    # BRCA -> i
    cfg$output_path <- gsub("Model_train/BRCA/", paste0("result_all_emb/Model_train/",i,"/"), cfg$output_path)
    
    # _OS_ -> _time_type_
    cfg$data$csv_path <- gsub("_OS_", paste0("_", time_type, "_"), cfg$data$csv_path)
    
    # geno_raw_dim
    cfg$data$csv_path <- gsub("BRCA", i, cfg$data$csv_path)
    cfg$models$geno_raw_dim <- as.integer(line_count + 1)
    cfg$data$geno_pt_path <- gsub("BRCA", i, cfg$data$geno_pt_path)
    cfg$data$tran_pt_path <- gsub("BRCA", i, cfg$data$tran_pt_path)
    # align_batch_size
    cfg$data$align_batch_size <-
      as.integer(as.numeric(batch_full_list[[i]]["align_batch_size_count"]))
    
    # prognosis lr（✅ 精确、无副作用）
    cfg$optimization$prognosis$lr <-
      as.numeric(batch_full_list[[i]]["lr_set"])
    cfg$optimization$t2i_weight_decay <-
      as.numeric(batch_full_list[[i]]["wd_set"])
    cfg$optimization$t2i_final_weight_decay <-
      as.numeric(batch_list[[i]]["wd_set"])
    
    # prognosis_batch_size
    cfg$data$prognosis_batch_size <-
      as.integer(as.numeric(batch_full_list[[i]]["prognosis_batch_size_count"]))
    
    cfg$optimization$align_epochs <-
      as.integer(as.numeric(batch_full_list[[i]][[time_type]]["align_epoch_num"]))
    
    cfg$optimization$prognosis_epochs <-
      as.integer(as.numeric(batch_full_list[[i]][[time_type]]["prognosis_epoch_num"]))
    
    old_scipen <- options("scipen")$scipen
    options(scipen = 999)
    # ===== 写回 =====
    write_yaml(cfg, temp_file)
    options(scipen = old_scipen)
    
    # 4. 移动临时文件到目标位置
    file.copy(temp_file, yaml_file, overwrite = TRUE)
    file.remove(temp_file)
    
    # 5. 验证结果
    if (file.exists(yaml_file) && file.info(yaml_file)$size > 0) {
      cat("✅ 配置文件生成成功: ", basename(yaml_file), "\n")
    } else {
      cat("❌ 配置文件生成失败: ", basename(yaml_file), "\n")
      next
    }
    
    # 6. 写入训练脚本
    if(time_type!="OS"){
      align_cmd <- sprintf("/home/Disk14T/chengpeng/miniconda3/envs/M3_JEPA/bin/python /home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/src/train_full_cohort_v5_3.py --fname %s --task align --run_name %s_%s",
                           yaml_file, i, time_type)
      prognosis_cmd <- sprintf("/home/Disk14T/chengpeng/miniconda3/envs/M3_JEPA/bin/python /home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/src/train_full_cohort_v5_3.py --fname %s --task prognosis --run_name %s_%s",
                               yaml_file, i, time_type)
      
      # =========================================================================
      # 💡 修改点 2: 将 align 和 prognosis 合并为一个串行任务组 (使用 && 符号)
      # =========================================================================
      group_cmd <- sprintf("%s && %s", align_cmd, prognosis_cmd)
      
      write(group_cmd, train_sh_file, append = TRUE)
    }
    
    # 7. 写入样本创建脚本
    
    
  }
}

# =========================================================================
# 💡 修改点 3: 闭合 heredoc 块，完成脚本编写
# =========================================================================
write("EOF", train_sh_file, append = TRUE)
write("echo \"🎉 所有模型训练任务执行完毕!\"", train_sh_file, append = TRUE)

cat("\n=== 处理完成 ===\n")



########################   2.2. full_cohort level inference

library(yaml)

input_path = "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_train/"
output_path ="/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_inference/Full_cohort/"
yaml_path <-"/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_inference/configs/Full_cohort/"

inference_sh_file <- paste0(output_path,"/../../2.2.all_cancer_inference_full_cohort.sh")
write("#!/bin/bash", inference_sh_file, append = FALSE)


for(each_cancer_type in list.files(input_path)){
  
  gene_file <- paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_genome/TCGA-", each_cancer_type, "_gene_list.txt")
  
  
  line_count <- as.integer(system(paste("wc -l <", shQuote(gene_file)), intern = TRUE))
  
  
  cancer_time <- list.files(paste0(input_path,"/",each_cancer_type))
  
  for(each_cancer_time in cancer_time){
    
    for(fold in c("full_cohort")){
      
      val_csv <-paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/",each_cancer_time,"_dataset.csv")
      align_pth <-paste0(input_path,"/",each_cancer_type,"/",each_cancer_time,"/",fold,"/align_full_final.pth.tar")
      prognosis_pth <-paste0(input_path,"/",each_cancer_type,"/",each_cancer_time,"/",fold,"/prognosis_full_final.pth.tar")
      
      for(each_test_mode in c("prognosis","align")){
        
        for( avi_modal_name  in c("Full_modal","Modal_miss_path","Modal_miss_text","Modal_miss_geno","Modal_miss_tran","Only_path","Only_text","Only_geno","Only_tran")){
          csv_path = val_csv
          path_pt_path = "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_SVS/TCGA_TITAN_first_slice_dict.pt"
          text_pt_path = "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/TCGA_data_clinical_patient_prompt.pt"
          geno_pt_path = paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_genome/TCGA-",each_cancer_type,".pt")
          tran_pt_path = paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_transcriptome/TCGA-",each_cancer_type,".pt")
          
          if(avi_modal_name=="Modal_miss_path"){
            path_pt_path = NULL
          }else if(avi_modal_name=="Modal_miss_text"){
            text_pt_path = NULL
          }else if(avi_modal_name=="Modal_miss_geno"){
            geno_pt_path = NULL
          }else if(avi_modal_name=="Modal_miss_tran"){
            tran_pt_path = NULL
          }else if(avi_modal_name=="Only_path"){
            tran_pt_path = NULL
            text_pt_path = NULL
            geno_pt_path = NULL
          }else if(avi_modal_name=="Only_text"){
            tran_pt_path = NULL
            path_pt_path = NULL
            geno_pt_path = NULL
          }else if(avi_modal_name=="Only_geno"){
            tran_pt_path = NULL
            text_pt_path = NULL
            path_pt_path = NULL
          }else if(avi_modal_name=="Only_tran"){
            path_pt_path = NULL
            text_pt_path = NULL
            geno_pt_path = NULL
          } else{
            NULL
          }
        
          run_name = fold
          each_output_path=paste0(output_path,"/",each_cancer_type,"/",each_cancer_time,"/",each_test_mode,"/",avi_modal_name,"/")
          each_model_path=paste0(input_path,"/",each_cancer_type,"/",each_cancer_time,"/",fold,"/",each_test_mode,"_full_final.pth.tar")
          each_yaml_path <-paste0(yaml_path,"/",each_cancer_type,"_",each_cancer_time,"_",each_test_mode,"_",avi_modal_name,"_",fold,".yaml") 
          


          
          yaml_data <- list(
            task_mode = each_test_mode,
            run_name = run_name,
            model_path = each_model_path,
            output_path = each_output_path,
            use_bfloat16 = TRUE,
            
            data = list(
              csv_path = csv_path,
              batch_size = 64L, # 使用 L 强制声明为整数，以保持格式整洁
              num_workers = 4L,
              pin_mem = TRUE,
              path_pt_path = path_pt_path,
              text_pt_path = text_pt_path,
              geno_pt_path = geno_pt_path,
              tran_pt_path = tran_pt_path
            ),
            
            models = list(
              text_encoder_name = "/home/Disk30T/chengpeng/Multimodal_JEPA/medgemma-1.5-4b-it",
              path_dim = 768L,
              geno_raw_dim = as.integer(line_count + 1),
              geno_dim = 512L,
              tran_dim = 643L
            ),
            
            MoE = list(
              num_experts = 8L,
              k = 2L,
              drop = 0.1,
              hidden_size = 1024L
            )
          )
          write_yaml(yaml_data, file = each_yaml_path)
          each_run <-paste0("/home/Disk14T/chengpeng/miniconda3/envs/M3_JEPA/bin/python /home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/src/inference_v5_3.py --fname " ,each_yaml_path)
          write(each_run, inference_sh_file, append = T)
          if(each_test_mode=="prognosis"){
            each_run <-paste0("/home/Disk14T/chengpeng/miniconda3/envs/M3_JEPA/bin/python /home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/src/evaluate_cindex.py --scores ", each_output_path,"/",run_name,"/prognosis_risk_scores.csv "," --yaml " ,each_yaml_path)
            write(each_run, inference_sh_file, append = T)
          }
        
        }
      
        
      }
    }
  }

}



########################   3. fold level unimodal train
train_sh_file <- "/home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/3.all_cancer_train_unimodal.sh"
template_file <- "/home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/configs/demo/train_BRCA_v3_5.yaml"

# =========================================================================
# 💡 修改点 1: 初始化 Bash 脚本，加入 xargs 并发控制框架
# =========================================================================
write("#!/bin/bash", train_sh_file, append = FALSE)
write("# 使用说明: bash 1.all_cancer_train_by_fold.sh [并发数量]", train_sh_file, append = TRUE)
write("# 如果不传参数，默认同时并行运行 4 组", train_sh_file, append = TRUE)
write("N_PARALLEL=${1:-1}", train_sh_file, append = TRUE)
write("echo \"🚀 开始执行训练任务，当前最大并行任务数: $N_PARALLEL\"", train_sh_file, append = TRUE)
write("echo \"---------------------------------------------------\"", train_sh_file, append = TRUE)
# 开始写入 heredoc，交由 xargs 调度
write("cat << 'EOF' | xargs -I {} -P $N_PARALLEL bash -c \"{}\"", train_sh_file, append = TRUE)

write("## iter run", csv_sh_file, append = FALSE)

for(time_type in c("OS","DSS","DFI","PFI")) {              
  for(i in c(cancer_type,"ALL")){
    
    cat("\n=== 处理: ", i, " - ", time_type, " ===\n", sep = "")
    
    yaml_file = paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/configs/unimodal/train_", i, "_", time_type, ".yaml")
    
    # 1. 检查模板文件
    if (!file.exists(template_file) || file.info(template_file)$size == 0) {
      stop("模板文件为空或不存在: ", template_file)
    }
    
    # 2. 获取基因数量
    gene_file <- paste0("/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_genome/TCGA-", i, "_gene_list.txt")
    if (!file.exists(gene_file)) {
      cat("⚠️  跳过 ", i, "，基因文件不存在\n", sep = "")
      next
    }
    
    line_count <- as.integer(system(paste("wc -l <", shQuote(gene_file)), intern = TRUE))
    cat("基因数量: ", line_count, "\n", sep = "")
    
    # 3. 使用临时文件处理
    temp_file <- tempfile(fileext = ".yaml")
    file.copy(template_file, temp_file, overwrite = TRUE)
    
    # 读取 YAML
    cfg <- read_yaml(temp_file)
    
    # ===== 一一对应你原来的 sed =====
    
    # BRCA -> i
    cfg$output_path <- gsub("Model_train", "Unimodal_train", cfg$output_path)
    cfg$output_path <- gsub("BRCA", i, cfg$output_path)
    
    # _OS_ -> _time_type_
    cfg$data$csv_path <- gsub("_OS_", paste0("_", time_type, "_"), cfg$data$csv_path)
    
    # geno_raw_dim
    cfg$data$csv_path <- gsub("BRCA", i, cfg$data$csv_path)
    cfg$models$geno_raw_dim <- as.integer(line_count + 1)
    cfg$data$geno_pt_path <- gsub("BRCA", i, cfg$data$geno_pt_path)
    cfg$data$tran_pt_path <- gsub("BRCA", i, cfg$data$tran_pt_path)
    # align_batch_size
    cfg$data$align_batch_size <-
      as.integer(as.numeric(batch_list[[i]]["align_batch_size_count"]))
    
    # prognosis lr（✅ 精确、无副作用）
    cfg$optimization$prognosis$lr <-
      as.numeric(batch_list[[i]]["lr_set"])
    cfg$optimization$t2i_weight_decay <-
      as.numeric(batch_list[[i]]["wd_set"])
    cfg$optimization$t2i_final_weight_decay <-
      as.numeric(batch_list[[i]]["wd_set"])
    # prognosis_batch_size
    cfg$data$prognosis_batch_size <-
      as.integer(as.numeric(batch_list[[i]]["prognosis_batch_size_count"]))
    
    
    
    old_scipen <- options("scipen")$scipen
    options(scipen = 999)
    # ===== 写回 =====
    write_yaml(cfg, temp_file)
    options(scipen = old_scipen)
    # 4. 移动临时文件到目标位置
    file.copy(temp_file, yaml_file, overwrite = TRUE)
    file.remove(temp_file)
    
    # 5. 验证结果
    if (file.exists(yaml_file) && file.info(yaml_file)$size > 0) {
      cat("✅ 配置文件生成成功: ", basename(yaml_file), "\n")
    } else {
      cat("❌ 配置文件生成失败: ", basename(yaml_file), "\n")
      next
    }
    
    # 6. 写入训练脚本
   # if(time_type=="OS"){

      prognosis_cmd <- sprintf("/home/Disk14T/chengpeng/miniconda3/envs/M3_JEPA/bin/python /home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA/run_unimodal.py --fname %s  --run_name %s_%s",
                               yaml_file, i, time_type)
      
      # =========================================================================
      # 💡 修改点 2: 将 align 和 prognosis 合并为一个串行任务组 (使用 && 符号)
      # =========================================================================

      
      write(prognosis_cmd, train_sh_file, append = TRUE)
   # }
    

    
  }
}

# =========================================================================
# 💡 修改点 3: 闭合 heredoc 块，完成脚本编写
# =========================================================================
write("EOF", train_sh_file, append = TRUE)
write("echo \"🎉 所有模型训练任务执行完毕!\"", train_sh_file, append = TRUE)

cat("\n=== 处理完成 ===\n")




###########  section2 #########


### 比较 单模态基线 vs 多模态融合 c-index
output_dir <-"/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section2/"

base_dir <- c("/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_inference/Split_by_fold/")
# 2. 获取所有推理结果的 CSV 文件路径
csv_files <- list.files(path = base_dir, 
                        pattern = "evaluation_metrics\\.csv$", 
                        recursive = TRUE, 
                        full.names = TRUE)

csv_files <-csv_files[grep("_OS",csv_files)]

# 3. 定义函数处理新结构的路径和 CSV
process_csv_new <- function(file_path) {
  # 逐层向上解析文件夹路径
  # 例如路径: .../Inference_result/SKCM/SKCM_OS/prognosis/Full_modal/fold_0/inference_c_index_result.csv
  dir_fold <- dirname(file_path)               # .../Full_modal/fold_0
  dir_modal <- dirname(dir_fold)               # .../prognosis/Full_modal
  dir_prog_fixed <- dirname(dir_modal)         # .../SKCM_OS/prognosis
  dir_cancer_prog <- dirname(dir_prog_fixed)   # .../SKCM/SKCM_OS
  dir_cancer <- dirname(dir_cancer_prog)       # .../Inference_result/SKCM
  
  # 提取所需的名称
  modal_name <- basename(dir_modal)            # 结果: "Full_modal" (或其他模态)
  cancer_prog_name <- basename(dir_cancer_prog)# 结果: "SKCM_OS"
  cancer_name <- basename(dir_cancer)          # 结果: "SKCM"
  
  # 提取预后类型 (把 "SKCM_OS" 中的 "SKCM_" 去掉，保留 "OS")
  prog_type <- str_replace(cancer_prog_name, paste0("^", cancer_name, "_"), "")
  
  # 读取 CSV，新结构为: Run_Name, C-index
  data <- read_csv(file_path, show_col_types = FALSE) %>%
    rename(Run_Name = 1, C_index = 2) %>%
    mutate(
      C_index = as.numeric(C_index),
      Cancer = cancer_name,
      Prognosis = prog_type,
      Modal = modal_name
    ) %>%
    # 只保留我们画图需要的列
    select(Cancer, Prognosis, Modal, Run_Name, C_index)
  
  return(data)
}

# 4. 批量执行并合并为长表
final_data_multi <- map_df(csv_files, process_csv_new)
final_data_multi$type<-"joint"

base_dir <- c("/home/Disk30T/chengpeng/Multimodal_JEPA/result/Unimodal_train/")

# 1. 获取所有推理结果的 CSV 文件路径
csv_files <- list.files(path = base_dir, 
                        pattern = "cv_results\\.csv$", 
                        recursive = TRUE, 
                        full.names = TRUE)

# 2. 定义函数处理新结构的路径和 CSV
process_csv_new <- function(file_path) {
  
  dir_modal <- dirname(file_path)              
  dir_cancer_prog <- dirname(dir_modal)        
  dir_cancer <- dirname(dir_cancer_prog)       
  
  modal_name <- basename(dir_modal)            
  cancer_prog_name <- basename(dir_cancer_prog)
  cancer_name <- basename(dir_cancer)          
  
  prog_type <- str_replace(cancer_prog_name, paste0("^", cancer_name, "_"), "")
  
  data <- read_csv(file_path, show_col_types = FALSE) %>%
    filter(!Fold %in% c("Average", "Std_Dev")) %>%
    # 【核心修改点】：使用排除法，只要不是 Fold 和 Best_Epoch，剩下的全转换
    pivot_longer(
      cols = -c(Fold, Best_Epoch), 
      names_to = "Run_Name", 
      values_to = "C_index"
    ) %>%
    mutate(
      Run_Name = case_when(
        Run_Name == "AUC_1yr" ~ "AUC_1-Year",
        Run_Name == "AUC_3yr" ~ "AUC_3-Year",
        Run_Name == "AUC_5yr" ~ "AUC_5-Year",
        # 增加一行：如果那个文件里叫 C_index，我们把它统一重命名为 C-index
        Run_Name == "C_index" ~ "C-index", 
        TRUE ~ Run_Name
      ),
      C_index = as.numeric(C_index),
      Cancer = cancer_name,
      Prognosis = prog_type,
      Modal = modal_name
    ) %>%
    select(Cancer, Prognosis, Modal, Run_Name, C_index)
  
  return(data)
}

# 3. 批量执行并合并为长表
final_data_single <- map_df(csv_files, process_csv_new)
final_data_single$type <-"raw"

final_data <-rbind(final_data_multi,final_data_single )
final_data <-subset(final_data,Run_Name!="Negative Val Loss")

summary_table <- subset(final_data,Modal %in%c("geno", "path", "text", "tran", "Mean_Pool_features","Concat_features","Prov_gigapath","Full_modal")) %>%
  # 1. [核心步骤] 将 Modal 转换为 factor，并严格指定我们想要的特定顺序
  mutate(Modal = factor(Modal, levels = c("geno", "path", "text", "tran","Mean_Pool_features","Concat_features","Prov_gigapath", "Full_modal"))) %>%
  
  # 2. 按 Modal（行）和 Cancer（列）分组
  group_by(Modal, Cancer) %>%
  
  # 3. 计算每个组别 C_index 的均值和标准差
  summarise(
    mean_val = mean(C_index, na.rm = TRUE),
    sd_val = sd(C_index, na.rm = TRUE),
    .groups = "drop"  
  ) %>%
  
  # 4. 拼接成 "均值 ± 标准差" 格式，强制保留 3 位小数
  mutate(
    formatted_value = sprintf("%.3f ± %.3f", mean_val, sd_val)
  ) %>%
  
  # 5. 只保留最终需要的列进行转换
  select(Modal, Cancer, formatted_value) %>%
  
  # 6. 长表转宽表：将 Cancer 铺开成列
  pivot_wider(
    names_from = Cancer,
    values_from = formatted_value
  ) %>%
  
  # 7. 根据之前设定的 factor levels 确保最终的行排序正确
  arrange(Modal)

write.csv(summary_table,paste0(output_dir, "S2_Baseline_singlemodal_vs_JEPA_fullmodal_cindex_table_", "OS", ".csv") ,fileEncoding = "GBK")


target_modals <- c("geno", "path", "text", "tran", "Full_modal","Mean_Pool_features","Concat_features","Prov_gigapath")

summary_auc_table <- final_data %>%
  # 1. 过滤掉不要的指标，只保留 AUC 相关的指标以及我们关注的模态
  filter(
    Modal %in% target_modals,
    Run_Name %in% c("iAUC", "AUC_1-Year", "AUC_3-Year", "AUC_5-Year")
  ) %>%
  
  # 2. 将 Modal 转换为 factor，严格指定行排序的顺序
  mutate(Modal = factor(Modal, levels = target_modals)) %>%
  
  # 3. [核心步骤] 按 Cancer、Modal 和 指标类型(Run_Name) 分组
  group_by(Cancer, Modal, Run_Name) %>%
  
  # 4. 计算每个指标的均值和标准差（注意：长表中你的数值列名依然叫 C_index）
  summarise(
    mean_val = mean(C_index, na.rm = TRUE),
    sd_val = sd(C_index, na.rm = TRUE),
    .groups = "drop"  
  ) %>%
  
  # 5. 拼接成 "均值 ± 标准差" 格式，强制保留 3 位小数
  mutate(
    formatted_value = sprintf("%.3f ± %.3f", mean_val, sd_val)
  ) %>%
  
  # 6. 只保留需要的列，准备长宽转换
  select(Cancer, Modal, Run_Name, formatted_value) %>%
  
  # 7. 长表转宽表：将 Run_Name（iAUC, AUC_1-Year等）铺开成列
  pivot_wider(
    names_from = Run_Name,
    values_from = formatted_value
  ) %>%
  
  # 8. 重命名列，使其完全符合你要求的最终展示格式
  rename(
    Models = Cancer,
    `AUC (1-year)` = `AUC_1-Year`,
    `AUC (3-year)` = `AUC_3-Year`,
    `AUC (5-year)` = `AUC_5-Year`
  ) %>%
  
  # 9. 确保需要的列都在，并按 Models 和指定的 Modal 顺序排列
  select(Models, Modal, iAUC, `AUC (1-year)`, `AUC (3-year)`, `AUC (5-year)`) %>%
  arrange(Models, Modal)


write.csv(summary_auc_table,paste0(output_dir, "S2_Baseline_singlemodal_vs_JEPA_fullmodal_auc_table_", "OS", ".csv") ,fileEncoding = "GBK")


#### 全模态，单一模态，缺失模态C-IDNEX组图

library(tidyverse)
library(patchwork)
library(ggsci) # 提供高分期刊配色


final_data <-final_data[-grep("train_riskscore",final_data$Modal),]


final_data$Modal <-as.character(final_data$Modal)
final_data$Modal[which(final_data$Modal=="Only_geno")] <-"OncoJEPA Gen-Only"
final_data$Modal[which(final_data$Modal=="Only_path")] <-"OncoJEPA Path-Only"
final_data$Modal[which(final_data$Modal=="Only_text")] <-"OncoJEPA Clin-Only"
final_data$Modal[which(final_data$Modal=="Only_tran")] <-"OncoJEPA Tran-Only"
final_data$Modal[which(final_data$Modal=="Modal_miss_geno")] <-"OncoJEPA w/o Gen"
final_data$Modal[which(final_data$Modal=="Modal_miss_path")] <-"OncoJEPA w/o Path"
final_data$Modal[which(final_data$Modal=="Modal_miss_text")] <-"OncoJEPA w/o Clin"
final_data$Modal[which(final_data$Modal=="Modal_miss_tran")] <-"OncoJEPA w/o Tran"
final_data$Modal[which(final_data$Modal=="Full_modal")] <-"OncoJEPA"

# 过滤特定指标
df_clean <- final_data %>% filter(Run_Name == "C-index", Prognosis == "OS") %>%
  # 清理 Modal 列中的冗余字符串
#  mutate(Modal = str_remove(Modal, "_train_riskscore")) %>%
  # 将 C_index 转换为数值型（以防原始数据被识别为字符）
  mutate(C_index = as.numeric(C_index))

# 1. 数据准备 (沿用之前的 df_clean)
# 定义模态分组
modal_full_miss <- c("OncoJEPA", "OncoJEPA w/o Tran", "OncoJEPA w/o Path", "OncoJEPA w/o Gen", "OncoJEPA w/o Clin")
modal_only <- c("OncoJEPA Gen-Only", "OncoJEPA Path-Only", "OncoJEPA Clin-Only", "OncoJEPA Tran-Only")

# 确保癌种因子化，并把 ALL 放在最前面
cancer_levels <- c("ALL", sort(unique(df_clean$Cancer[df_clean$Cancer != "ALL"])))
df_clean$Cancer <- factor(df_clean$Cancer, levels = cancer_levels)

# 获取 ALL 队列中 Full_modal 的中位数，作为全局参考线
global_baseline <- median(df_clean$C_index[df_clean$Cancer == "ALL" & df_clean$Modal == "OncoJEPA"], na.rm = TRUE)

# 2. 绘制上图 (Panel A)
data_top <- df_clean %>% 
  filter(Modal %in% modal_full_miss) %>%
  mutate(Modal = factor(Modal, levels = modal_full_miss))

plot_top <- ggplot(data_top, aes(x = Cancer, y = C_index, fill = Modal)) +
  geom_boxplot(width = 0.7, alpha = 0.9, outlier.size = 0.3, linewidth = 0.3) +
  # 添加全局基准线和分割 ALL 的竖线
  geom_hline(yintercept = global_baseline, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = 1.5, linetype = "dashed", color = "black", linewidth = 0.8) +
  scale_fill_npg() + # 使用 Nature 风格配色
  theme_bw(base_size = 7) +
  labs(title = "A. Robustness to Missing Modalities across Pan-Cancer Landscape", x = "", y = "C-index") +
  theme(
    axis.text.x = element_blank(), # 隐藏上图的 X 轴标签，由下图统一显示
    axis.ticks.x = element_blank(),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    panel.grid.major.x = element_blank(),
    plot.title = element_text(face = "bold")
  )

# 3. 绘制下图 (Panel B)
data_bottom <- df_clean %>% 
  filter(Modal %in% modal_only) %>%
  mutate(Modal = factor(Modal, levels = modal_only))

plot_bottom <- ggplot(data_bottom, aes(x = Cancer, y = C_index, fill = Modal)) +
  geom_boxplot(width = 0.7, alpha = 0.9, outlier.size = 0.3, linewidth = 0.3) +
  # 同样添加竖线分割 ALL
  geom_vline(xintercept = 1.5, linetype = "dashed", color = "black", linewidth = 0.8) +
  # 下图可以用另一套深邃一点的配色，或者自定义颜色区分开
  scale_fill_d3() + 
  theme_bw(base_size = 7) +
  # 为了方便对比，建议上下图 Y 轴范围大致一致或相近
  coord_cartesian(ylim = c(min(df_clean$C_index, na.rm=T), max(df_clean$C_index, na.rm=T))) + 
  labs(title = "B. Single Modality Contributions (No Modality dominant)", x = "Cancer Cohorts", y = "C-index") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    panel.grid.major.x = element_blank(),
    plot.title = element_text(face = "bold")
  )

# 4. 完美拼接 (使用 patchwork)
# 强制上下对齐坐标轴
final_landscape <- plot_top / plot_bottom + plot_layout(axes = "collect")

# 导出高质量长图
ggsave("/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section2/S2_JEPA_cindex_High_Level_Landscape.pdf", final_landscape, width = 6, height = 4, dpi = 300)



summary_table <- subset(final_data,Modal %in%c("OncoJEPA w/o Gen", "OncoJEPA w/o Path", "OncoJEPA w/o Clin", "OncoJEPA w/o Tran","OncoJEPA Path-Only", "OncoJEPA Tran-Only", "OncoJEPA Gen-Only", "OncoJEPA Clin-Only", "OncoJEPA")) %>%
  # 1. [核心步骤] 将 Modal 转换为 factor，并严格指定我们想要的特定顺序
  mutate(Modal = factor(Modal, levels = c("OncoJEPA w/o Gen", "OncoJEPA w/o Path", "OncoJEPA w/o Clin", "OncoJEPA w/o Tran", "OncoJEPA Path-Only", "OncoJEPA Tran-Only", "OncoJEPA Gen-Only", "OncoJEPA Clin-Only","OncoJEPA"))) %>%
  
  # 2. 按 Modal（行）和 Cancer（列）分组
  group_by(Modal, Cancer) %>%
  
  # 3. 计算每个组别 C_index 的均值和标准差
  summarise(
    mean_val = mean(C_index, na.rm = TRUE),
    sd_val = sd(C_index, na.rm = TRUE),
    .groups = "drop"  
  ) %>%
  
  # 4. 拼接成 "均值 ± 标准差" 格式，强制保留 3 位小数
  mutate(
    formatted_value = sprintf("%.3f ± %.3f", mean_val, sd_val)
  ) %>%
  
  # 5. 只保留最终需要的列进行转换
  select(Modal, Cancer, formatted_value) %>%
  
  # 6. 长表转宽表：将 Cancer 铺开成列
  pivot_wider(
    names_from = Cancer,
    values_from = formatted_value
  ) %>%
  
  # 7. 根据之前设定的 factor levels 确保最终的行排序正确
  arrange(Modal)



write.csv(summary_table,paste0(output_dir, "S2_OncoJEPA_full_vs_miss_cindex_table_", "OS", ".csv") ,fileEncoding = "GBK")

extract_mean <- function(x) as.numeric(str_extract(x, "^[0-9.]+"))
extract_sd   <- function(x) as.numeric(str_extract(x, "±\\s*([0-9.]+)", group = 1))

# 筛选行
onco_row  <- summary_table %>% filter(as.character(Modal) == "OncoJEPA")
wo_rows   <- summary_table %>% filter(str_detect(as.character(Modal), "^OncoJEPA w/o"))
only_rows <- summary_table %>% filter(str_detect(as.character(Modal), "Only$"))  # 或以 "^OncoJEPA.*Only$" 更精确

# 其他列名（排除 Modal 和 ALL）
other_cols <- setdiff(names(summary_table), c("Modal", "ALL"))

# ----- 计算 -----

# ① ALL 列中 OncoJEPA 的值（原始字符串）
all_onco_raw <- onco_row$ALL

# ② ALL 列中 OncoJEPA w/o 行的均值 ± 标准差
all_wo_means <- extract_mean(wo_rows$ALL)
all_wo_mean  <- mean(all_wo_means)
all_wo_sd    <- sd(all_wo_means)
all_wo_out   <- sprintf("%.4f ± %.4f", all_wo_mean, all_wo_sd)

# ③ 其他列中 OncoJEPA 行的均值 ± 标准差
others_onco_values <- onco_row %>%
  select(all_of(other_cols)) %>%
  unlist() %>%
  extract_mean()
others_onco_mean <- mean(others_onco_values)
others_onco_sd   <- sd(others_onco_values)
others_onco_out  <- sprintf("%.4f ± %.4f", others_onco_mean, others_onco_sd)

# ④ 其他列中 OncoJEPA w/o 行的均值 ± 标准差
others_wo_values <- wo_rows %>%
  select(all_of(other_cols)) %>%
  unlist() %>%
  extract_mean()
others_wo_mean <- mean(others_wo_values)
others_wo_sd   <- sd(others_wo_values)
others_wo_out  <- sprintf("%.4f ± %.4f", others_wo_mean, others_wo_sd)

# ⑤ ALL 列中 Only 行的均值 ± 标准差
all_only_means <- extract_mean(only_rows$ALL)
all_only_mean  <- mean(all_only_means)
all_only_sd    <- sd(all_only_means)
all_only_out   <- sprintf("%.4f ± %.4f", all_only_mean, all_only_sd)

# ⑥ 其他列中 Only 行的均值 ± 标准差
others_only_values <- only_rows %>%
  select(all_of(other_cols)) %>%
  unlist() %>%
  extract_mean()
others_only_mean <- mean(others_only_values)
others_only_sd   <- sd(others_only_values)
others_only_out  <- sprintf("%.4f ± %.4f", others_only_mean, others_only_sd)

# ----- 输出结果 -----
cat("=== 计算结果 ===\n")
cat("① ALL 列中 OncoJEPA 的值：", all_onco_raw, "\n")
cat("② ALL 列中 OncoJEPA w/o 行的均值 ± 标准差：", all_wo_out, "\n")
cat("③ 其他列中 OncoJEPA 行的均值 ± 标准差：", others_onco_out, "\n")
cat("④ 其他列中 OncoJEPA w/o 行的均值 ± 标准差：", others_wo_out, "\n")
cat("⑤ ALL 列中 Only 行的均值 ± 标准差：", all_only_out, "\n")
cat("⑥ 其他列中 Only 行的均值 ± 标准差：", others_only_out, "\n")

#① ALL 列中 OncoJEPA 的值： 0.782 ± 0.017 
#② ALL 列中 OncoJEPA w/o 行的均值 ± 标准差： 0.7742 ± 0.0081 
#③ 其他列中 OncoJEPA 行的均值 ± 标准差： 0.7191 ± 0.0688 
#④ 其他列中 OncoJEPA w/o 行的均值 ± 标准差： 0.6924 ± 0.0684 
#⑤ ALL 列中 Only 行的均值 ± 标准差： 0.7215 ± 0.0595 
#⑥ 其他列中 Only 行的均值 ± 标准差： 0.6110 ± 0.0955

######  only joint emb c-index plot

output_dir <-"/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section2/"

base_dir <- c("/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_only_joint_emb/Model_inference/Split_by_fold/")
# 2. 获取所有推理结果的 CSV 文件路径
csv_files <- list.files(path = base_dir, 
                        pattern = "evaluation_metrics\\.csv$", 
                        recursive = TRUE, 
                        full.names = TRUE)

csv_files <-csv_files[grep("_OS",csv_files)]

# 3. 定义函数处理新结构的路径和 CSV
process_csv_new <- function(file_path) {
  # 逐层向上解析文件夹路径
  # 例如路径: .../Inference_result/SKCM/SKCM_OS/prognosis/Full_modal/fold_0/inference_c_index_result.csv
  dir_fold <- dirname(file_path)               # .../Full_modal/fold_0
  dir_modal <- dirname(dir_fold)               # .../prognosis/Full_modal
  dir_prog_fixed <- dirname(dir_modal)         # .../SKCM_OS/prognosis
  dir_cancer_prog <- dirname(dir_prog_fixed)   # .../SKCM/SKCM_OS
  dir_cancer <- dirname(dir_cancer_prog)       # .../Inference_result/SKCM
  
  # 提取所需的名称
  modal_name <- basename(dir_modal)            # 结果: "Full_modal" (或其他模态)
  cancer_prog_name <- basename(dir_cancer_prog)# 结果: "SKCM_OS"
  cancer_name <- basename(dir_cancer)          # 结果: "SKCM"
  
  # 提取预后类型 (把 "SKCM_OS" 中的 "SKCM_" 去掉，保留 "OS")
  prog_type <- str_replace(cancer_prog_name, paste0("^", cancer_name, "_"), "")
  
  # 读取 CSV，新结构为: Run_Name, C-index
  data <- read_csv(file_path, show_col_types = FALSE) %>%
    rename(Run_Name = 1, C_index = 2) %>%
    mutate(
      C_index = as.numeric(C_index),
      Cancer = cancer_name,
      Prognosis = prog_type,
      Modal = modal_name
    ) %>%
    # 只保留我们画图需要的列
    select(Cancer, Prognosis, Modal, Run_Name, C_index)
  
  return(data)
}

final_data_multi <- map_df(csv_files, process_csv_new)


final_data <-final_data_multi
final_data <-subset(final_data,Run_Name!="Negative Val Loss")

final_data <-final_data[-grep("train_riskscore",final_data$Modal),]

final_data$Modal <-as.character(final_data$Modal)
final_data$Modal[which(final_data$Modal=="Modal_miss_geno")] <-"Joint emb w/o Gen"
final_data$Modal[which(final_data$Modal=="Modal_miss_path")] <-"Joint emb w/o Path"
final_data$Modal[which(final_data$Modal=="Modal_miss_text")] <-"Joint emb w/o Clin"
final_data$Modal[which(final_data$Modal=="Modal_miss_tran")] <-"Joint emb w/o Tran"
final_data$Modal[which(final_data$Modal=="Full_modal")] <-"Joint emb"


df_clean <- final_data %>% filter(Run_Name == "C-index", Prognosis == "OS") %>%
  # 清理 Modal 列中的冗余字符串
  #  mutate(Modal = str_remove(Modal, "_train_riskscore")) %>%
  # 将 C_index 转换为数值型（以防原始数据被识别为字符）
  mutate(C_index = as.numeric(C_index))

# 1. 数据准备 (沿用之前的 df_clean)
# 定义模态分组
modal_full_miss <- c("Joint emb", "Joint emb w/o Path", "Joint emb w/o Tran", "Joint emb w/o Gen", "Joint emb w/o Clin")

# 确保癌种因子化，并把 ALL 放在最前面
cancer_levels <- c("ALL", sort(unique(df_clean$Cancer[df_clean$Cancer != "ALL"])))
df_clean$Cancer <- factor(df_clean$Cancer, levels = cancer_levels)

# 获取 ALL 队列中 Full_modal 的中位数，作为全局参考线
global_baseline <- median(df_clean$C_index[df_clean$Cancer == "ALL" & df_clean$Modal == "Joint emb"], na.rm = TRUE)

# 2. 绘制上图 (Panel A)
data_top <- df_clean %>% 
  filter(Modal %in% modal_full_miss) %>%
  mutate(Modal = factor(Modal, levels = modal_full_miss))

plot_top <- ggplot(data_top, aes(x = Cancer, y = C_index, fill = Modal)) +
  geom_boxplot(width = 0.7, alpha = 0.9, outlier.size = 0.3, linewidth = 0.3) +
  # 添加全局基准线和分割 ALL 的竖线
  geom_hline(yintercept = global_baseline, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = 1.5, linetype = "dashed", color = "black", linewidth = 0.8) +
  scale_fill_npg() + # 使用 Nature 风格配色
  theme_bw(base_size = 7) +
  labs(title = "A. Robustness to Missing Modalities across Pan-Cancer Landscape", x = "", y = "C-index") +
  theme(
   # axis.text.x = element_blank(), # 隐藏上图的 X 轴标签，由下图统一显示
    axis.ticks.x = element_blank(),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    panel.grid.major.x = element_blank(),
    plot.title = element_text(face = "bold")
  )

# 导出高质量长图
ggsave("/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section2/S2_JEPA_onlyjointemb_cindex_High_Level_Landscape.pdf", plot_top, width = 6, height = 2, dpi = 300)

summary_table <- subset(final_data,Modal %in%c("Joint emb w/o Gen", "Joint emb w/o Path", "Joint emb w/o Clin", "Joint emb w/o Tran", "Joint emb")) %>%
  # 1. [核心步骤] 将 Modal 转换为 factor，并严格指定我们想要的特定顺序
  mutate(Modal = factor(Modal, levels = c("Joint emb w/o Gen", "Joint emb w/o Path", "Joint emb w/o Clin", "Joint emb w/o Tran", "Joint emb"))) %>%
  
  # 2. 按 Modal（行）和 Cancer（列）分组
  group_by(Modal, Cancer) %>%
  
  # 3. 计算每个组别 C_index 的均值和标准差
  summarise(
    mean_val = mean(C_index, na.rm = TRUE),
    sd_val = sd(C_index, na.rm = TRUE),
    .groups = "drop"  
  ) %>%
  
  # 4. 拼接成 "均值 ± 标准差" 格式，强制保留 3 位小数
  mutate(
    formatted_value = sprintf("%.3f ± %.3f", mean_val, sd_val)
  ) %>%
  
  # 5. 只保留最终需要的列进行转换
  select(Modal, Cancer, formatted_value) %>%
  
  # 6. 长表转宽表：将 Cancer 铺开成列
  pivot_wider(
    names_from = Cancer,
    values_from = formatted_value
  ) %>%
  
  # 7. 根据之前设定的 factor levels 确保最终的行排序正确
  arrange(Modal)



write.csv(summary_table,paste0(output_dir, "S2_Joint_emb_full_vs_miss_cindex_table_", "OS", ".csv") ,fileEncoding = "GBK")


# 辅助函数（与之前相同）
extract_mean <- function(x) as.numeric(str_extract(x, "^[0-9.]+"))
extract_sd   <- function(x) as.numeric(str_extract(x, "±\\s*([0-9.]+)", group = 1))

# 筛选行
joint_row <- summary_table %>% filter(as.character(Modal) == "Joint emb")
wo_rows   <- summary_table %>% filter(str_detect(as.character(Modal), "^Joint emb w/o"))

# 其他列名（排除 Modal 和 ALL）
other_cols <- setdiff(names(summary_table), c("Modal", "ALL"))

# ----- 计算 -----

# ① ALL 列中 Joint emb 的值（原始字符串）
all_joint_raw <- joint_row$ALL

# ② ALL 列中 Joint emb w/o 行的均值 ± 标准差
all_wo_means <- extract_mean(wo_rows$ALL)
all_wo_mean  <- mean(all_wo_means)
all_wo_sd    <- sd(all_wo_means)
all_wo_out   <- sprintf("%.4f ± %.4f", all_wo_mean, all_wo_sd)

# ③ 其他列中 Joint emb 行的均值 ± 标准差
others_joint_values <- joint_row %>%
  select(all_of(other_cols)) %>%
  unlist() %>%
  extract_mean()
others_joint_mean <- mean(others_joint_values)
others_joint_sd   <- sd(others_joint_values)
others_joint_out  <- sprintf("%.4f ± %.4f", others_joint_mean, others_joint_sd)

# ④ 其他列中 Joint emb w/o 行的均值 ± 标准差
others_wo_values <- wo_rows %>%
  select(all_of(other_cols)) %>%
  unlist() %>%
  extract_mean()
others_wo_mean <- mean(others_wo_values)
others_wo_sd   <- sd(others_wo_values)
others_wo_out  <- sprintf("%.4f ± %.4f", others_wo_mean, others_wo_sd)

# ----- 输出结果 -----
cat("=== Joint emb 计算结果 ===\n")
cat("① ALL 列中 Joint emb 的值：", all_joint_raw, "\n")
cat("② ALL 列中 Joint emb w/o 行的均值 ± 标准差：", all_wo_out, "\n")
cat("③ 其他列中 Joint emb 行的均值 ± 标准差：", others_joint_out, "\n")
cat("④ 其他列中 Joint emb w/o 行的均值 ± 标准差：", others_wo_out, "\n")

#① ALL 列中 Joint emb 的值： 0.760 ± 0.028 
#② ALL 列中 Joint emb w/o 行的均值 ± 标准差： 0.7568 ± 0.0035 
#③ 其他列中 Joint emb 行的均值 ± 标准差： 0.6881 ± 0.0585 
#④ 其他列中 Joint emb w/o 行的均值 ± 标准差： 0.6674 ± 0.0571

###########  KM曲线
library(tidyverse)
library(survival)
library(survminer)

base_dir <- "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_inference/Split_by_fold/"
output_dir <- "/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section2/"
# 新增：定义训练阶段输出目录（包含真实临床标签的 val.csv）
train_base_dir <- "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_train/" 

if(!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

surv_types <- "OS" # 当前只跑 OS
folds <- 0:4

time_col <- "survival_time"   # val.csv 中的存活时间列名
event_col <- "survival_event" # val.csv 中的生存结局列名
risk_col <- "risk_score"      # 预测分数的主列名

cancer_types <- list.dirs(base_dir, full.names = FALSE, recursive = FALSE)

# ================= 2. 核心分析循环 =================
for (surv in surv_types) {
  cat(sprintf("\n🚀 开始处理生存类型: %s\n", surv))
  plot_list <- list()
  
  for (cancer in cancer_types) {
    pooled_val_data <- data.frame()
    valid_cancer <- TRUE
    
    for (fold in folds) {
      # 1. 预测结果路径 (风险分数)
      train_dir <- file.path(base_dir, cancer, paste0(cancer, "_", surv), "prognosis", "Full_modal_train_riskscore", paste0("fold_", fold))
      train_file <- file.path(train_dir, "prognosis_risk_scores.csv")
      
      val_dir <- file.path(base_dir, cancer, paste0(cancer, "_", surv), "prognosis", "Full_modal", paste0("fold_", fold))
      val_file <- file.path(val_dir, "prognosis_risk_scores.csv")
      
      # 2. 新增：真实的临床结局数据路径 (来自 Model_train 目录)
      val_clinical_file <- file.path(train_base_dir, cancer, paste0(cancer, "_", surv), paste0("fold_", fold), "val.csv")
      
      # 检查三个文件是否都存在
      if (!file.exists(train_file) || !file.exists(val_file) || !file.exists(val_clinical_file)) {
        cat(sprintf("⚠️ 找不到 %s 的 %s fold_%d 的完整数据 (包含临床标签)，跳过该折...\n", cancer, surv, fold))
        valid_cancer <- FALSE
        break
      }
      
      #读取预测结果
      train_df <- read_csv(train_file, show_col_types = FALSE)
      colnames(train_df) <- c("sample_id", risk_col)
      
      val_df <- read_csv(val_file, show_col_types = FALSE)
      colnames(val_df) <- c("sample_id", risk_col)
      
      # 读取真实的临床结局数据
      val_clinical_df <- read_csv(val_clinical_file, show_col_types = FALSE)
      
      # ⭐ 核心修改：将真实标签与验证集的预测风险分数根据 sample_id 进行合并
      # 前提是 val_clinical_df 里面也有一个可以用来匹配的样本 ID 列。
      # 如果 val_clinical_df 中的 ID 列名不是 sample_id（例如是 ID 或 case_id），
      # 请修改这里，例如使用：by = c("sample_id" = "case_id")
      val_merged <- val_df %>%
        inner_join(val_clinical_df, by = "sample_id")
      
      # 获取训练集的风险分数中位数
      train_median <- median(train_df[[risk_col]], na.rm = TRUE)
      
      # 拿训练集截点切分合并后的验证集
      val_final <- val_merged %>%
        mutate(
          Risk_Group = ifelse(.data[[risk_col]] > train_median, "High Risk", "Low Risk")
        ) %>%
        select(all_of(c(time_col, event_col)), Risk_Group)
      
      # 追加到池化数据框中
      pooled_val_data <- bind_rows(pooled_val_data, val_final)
    }
    
    if (!valid_cancer || nrow(pooled_val_data) == 0) {
      next 
    }
    
    pooled_val_data$Risk_Group <- factor(pooled_val_data$Risk_Group, levels = c("Low Risk", "High Risk"))
    
    formula_str <- paste0("Surv(", time_col, ", ", event_col, ") ~ Risk_Group")
    fit <- surv_fit(as.formula(formula_str), data = pooled_val_data)
    
    p <- ggsurvplot(
      fit,
      data = pooled_val_data,
      pval = TRUE,               
      pval.coord = c(0, 0.1),    
      conf.int = TRUE,           
      risk.table = FALSE,        
      title = paste0(cancer, " (", surv, ")"), 
      palette = c("#1f77b4", "#d62728"),      
      legend.title = "Prediction",
      legend.labs = c("Low Risk", "High Risk"),
      ggtheme = theme_classic() + theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
        legend.position = "none" 
      )
    )
    
    plot_list[[cancer]] <- p
  }
  
  # ================= 3. 输出大图 =================
  if (length(plot_list) > 0) {
    cat(sprintf("✅ 成功生成 %s 的 %d 个癌种曲线，正在拼图输出...\n", surv, length(plot_list)))
    
    num_plots <- length(plot_list)
    n_cols <- min(7, num_plots) 
    n_rows <- ceiling(num_plots / n_cols)
    
    out_pdf <- file.path(output_dir, paste0("/S2_risk_score_group_KM_SurvType_", surv, "_PanCancer.pdf"))
    pdf(out_pdf, width = n_cols * 3.5, height = n_rows * 3.5)
    
    # print = TRUE 会将排版结果直接绘制到当前打开的 PDF 设备上
    res <- arrange_ggsurvplots(
      plot_list, 
      print = TRUE,          # 关键：改为 TRUE
      ncol = n_cols, 
      nrow = n_rows
    )
    
    dev.off() 
    cat(sprintf("🎉 保存成功: %s\n", out_pdf))
  }
}
cat("\n🏁 所有生存类型的分析绘图已全部完成！\n")


##########  基于risk score 绘制森林图
library(survival)
library(survminer)
library(dplyr)
library(readr)
library(stringr)
library(ggplot2)

# ================= 1. 路径与参数配置 =================
base_dir <- "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_inference/Split_by_fold"
train_base_dir <- "/home/Disk30T/chengpeng/Multimodal_JEPA/result/result_all_emb/Model_train" 
clinical_file <- "/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/Survival_SupplementalTable_S1_20171025_xena_sp"
output_dir <- "/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section2/"

if(!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

target_surv <- "OS"          
folds <- 0:4

id_col <- "sample_id"        
time_col <- "survival_time"  
event_col <- "survival_event"
risk_col <- "risk_score" 

# 获取所有癌种列表
cancer_types <- list.dirs(base_dir, full.names = FALSE, recursive = FALSE)

# ================= 2. 仅读取并清洗一次 TCGA 临床数据 =================
cat("⏳ 读取并清洗 TCGA 临床表型数据 (仅执行一次)...\n")
clin_df <- read_tsv(clinical_file, show_col_types = FALSE)

clin_df <- clin_df %>%
  mutate(patient_id = substr(sample, 1, 15)) %>% 
  select(patient_id, age_at_initial_pathologic_diagnosis, gender, ajcc_pathologic_tumor_stage) %>%
  distinct(patient_id, .keep_all = TRUE) %>%
  mutate(
    Age = as.numeric(age_at_initial_pathologic_diagnosis),
    Gender = str_to_title(gender),
    Stage = case_when(
      str_detect(ajcc_pathologic_tumor_stage, "(?i)Stage IV") ~ "Stage IV",
      str_detect(ajcc_pathologic_tumor_stage, "(?i)Stage III") ~ "Stage III",
      str_detect(ajcc_pathologic_tumor_stage, "(?i)Stage II") ~ "Stage II",
      str_detect(ajcc_pathologic_tumor_stage, "(?i)Stage I") ~ "Stage I",
      TRUE ~ NA_character_ 
    )
  ) %>%
  filter(!is.na(Age), !is.na(Gender), !is.na(Stage)) %>%
  select(patient_id, Age, Gender, Stage)

# ================= 3. 开启多页 PDF 绘图设备 =================
out_pdf <- file.path(output_dir, paste0("/S2_PanCancer_Multivariate_Cox_Forest_", target_surv, ".pdf"))
pdf(file = out_pdf, width = 8, height = 6, onefile = TRUE) # onefile = TRUE 允许合并多页
cat(sprintf("\n📄 已创建多页 PDF 文档准备写入: %s\n", out_pdf))

# ================= 4. 泛癌种循环分析 =================
for (target_cancer in cancer_types) {
  cat(sprintf("\n🚀 正在处理癌种: %s (%s)...\n", target_cancer, target_surv))
  
  pooled_ai_data <- data.frame()
  
  # 4.1 获取 5 折交叉验证数据
  for (fold in folds) {
    train_risk_file <- file.path(base_dir, target_cancer, paste0(target_cancer, "_", target_surv), "prognosis", "Full_modal_train_riskscore", paste0("fold_", fold), "prognosis_risk_scores.csv")
    val_risk_file <- file.path(base_dir, target_cancer, paste0(target_cancer, "_", target_surv), "prognosis", "Full_modal", paste0("fold_", fold), "prognosis_risk_scores.csv")
    val_clinical_file <- file.path(train_base_dir, target_cancer, paste0(target_cancer, "_", target_surv), paste0("fold_", fold), "val.csv")
    
    if (file.exists(train_risk_file) && file.exists(val_risk_file) && file.exists(val_clinical_file)) {
      train_df <- read_csv(train_risk_file, show_col_types = FALSE)
      colnames(train_df) <- c("sample_id", risk_col)
      train_median <- median(train_df[[risk_col]], na.rm = TRUE)
      
      val_df <- read_csv(val_risk_file, show_col_types = FALSE)
      colnames(val_df) <- c("sample_id", risk_col)
      
      val_clin_df <- read_csv(val_clinical_file, show_col_types = FALSE)
      val_merged <- inner_join(val_df, val_clin_df, by = "sample_id")
      
      val_final <- val_merged %>%
        mutate(
          Risk_Group = ifelse(.data[[risk_col]] > train_median, "High Risk", "Low Risk"),
          patient_id = substr(sample_id, 1, 15) 
        ) %>%
        select(patient_id, all_of(c(time_col, event_col)), Risk_Group)
      
      pooled_ai_data <- bind_rows(pooled_ai_data, val_final)
    }
  }
  
  if (nrow(pooled_ai_data) == 0) {
    cat(sprintf("⚠️ %s 无可用数据，跳过。\n", target_cancer))
    next
  }
  
  # 4.2 融合临床协变量与因子化
  final_merged_data <- inner_join(pooled_ai_data, clin_df, by = "patient_id")
  
  # 检查合并后是否有足够的数据拟合模型
  if(nrow(final_merged_data) < 30) {
    cat(sprintf("⚠️ %s 合并临床表型后样本量过少 (%d)，跳过回归。\n", target_cancer, nrow(final_merged_data)))
    next
  }
  
  final_merged_data <- final_merged_data %>%
    mutate(
      Risk_Group = factor(Risk_Group, levels = c("Low Risk", "High Risk")), 
      Stage = factor(Stage, levels = c("Stage I", "Stage II", "Stage III", "Stage IV")), 
      Gender = factor(Gender),
      Age_Group = ifelse(Age >= 65, ">=65", "<65"),
      Age_Group = factor(Age_Group, levels = c("<65", ">=65"))
    )
  
  # 4.3 🤖 智能构建 Cox 公式 (解决单性别报错)
  # 基础协变量
  covariates <- c("Risk_Group", "Age_Group", "Stage")
  
  # 如果队列中有男性也有女性，才将 Gender 加入协变量
  if (length(unique(na.omit(final_merged_data$Gender))) > 1) {
    covariates <- c(covariates, "Gender")
  } else {
    cat(sprintf("💡 提示: %s 为单性别或性别极度不平衡队列，已自动剔除 Gender 变量以防报错。\n", target_cancer))
  }
  
  formula_str <- paste0("Surv(", time_col, ", ", event_col, ") ~ ", paste(covariates, collapse = " + "))
  
  # 4.4 拟合并画图写入 PDF
  tryCatch({
    res.cox <- coxph(as.formula(formula_str), data = final_merged_data)
    
    forest_plot <- ggforest(
      res.cox, 
      data = final_merged_data, 
      main = paste0(target_cancer, " Multivariate Cox Regression (", target_surv, ")"),
      fontsize = 1.0, 
      noDigits = 2
    )
    
    # 打印当前图像到之前开启的 pdf 设备中 (新开一页)
    print(forest_plot)
    cat(sprintf("✅ %s 森林图已写入。\n", target_cancer))
    
  }, error = function(e) {
    cat(sprintf("❌ %s Cox 回归拟合或画图失败: %s\n", target_cancer, e$message))
  })
}

# ================= 5. 安全关闭 PDF 绘图设备 =================
dev.off()
cat(sprintf("\n🎉 所有森林图已成功汇总保存至: %s\n", out_pdf))


#######  CPTAC双模态数据集 C-INDEX 柱状图

library(ggplot2)
library(dplyr)
library(readr)
library(stringr)

# 1. 基础路径设置
base_dir <- "/home/Disk30T/chengpeng/Multimodal_JEPA/result/Model_inference/External_validation/"

# 2. 自动获取所有包含 evaluation_metrics.csv 的文件路径
# recursive = TRUE 允许搜索所有子文件夹
csv_files <- list.files(path = base_dir, 
                        pattern = "evaluation_metrics\\.csv$", 
                        recursive = TRUE, 
                        full.names = TRUE)

# 仅保留在 "_External_validation" 文件夹下的目标文件
csv_files <- csv_files[grepl("_External_validation/evaluation_metrics\\.csv$", csv_files)]

# 初始化一个空的数据框来存储结果
results <- data.frame(Cancer = character(), C_index = numeric(), stringsAsFactors = FALSE)

# 3. 遍历文件，提取数据
for (file_path in csv_files) {
  # 提取文件夹名，例如 UCEC_External_validation
  folder_name <- basename(dirname(file_path))
  
  # 提取癌症缩写并按要求拼接为 CPTAC-癌症名
  cancer_name <- str_replace(folder_name, "_External_validation", "")
  plot_name <- paste0("CPTAC-", cancer_name)
  
  # 尝试读取并处理数据
  tryCatch({
    # 读取csv，忽略列类型提示以保持终端整洁
    df <- read_csv(file_path, show_col_types = FALSE)
    
    # 筛选 Metric 为 "C-index" 的行并提取对应数值
    c_index_val <- df %>%
      filter(Metric == "C-index") %>%
      pull(Value) %>%
      as.numeric()
    
    # 如果成功提取到数值，则存入结果中
    if (length(c_index_val) > 0) {
      results <- rbind(results, data.frame(Cancer = plot_name, C_index = c_index_val[1]))
    }
  }, error = function(e) {
    message(paste("读取或解析文件时出错:", file_path))
  })
}

# 4. 绘图
if (nrow(results) == 0) {
  print("没有找到任何数据，请检查路径或文件格式是否正确。")
} else {
  # 按照 C-index 的值从高到低对 X 轴因子进行排序，使柱状图呈降序排列
  results$Cancer <- factor(results$Cancer, levels = results$Cancer[order(-results$C_index)])
  
  # 使用 ggplot2 绘制柱状图
  p <- ggplot(results, aes(x = Cancer, y = C_index)) +
    # 绘制柱子，设置颜色和边框
    geom_bar(stat = "identity", fill = "#4C72B0", color = "black", width = 0.5) +
    # 在正上方添加具体数值（保留四位小数）
    geom_text(aes(label = sprintf("%.4f", C_index)), vjust = -0.5, size = 4.5) +
    # 添加 y=0.5 的随机参考线
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray50", linewidth = 0.8) +
    # 设置Y轴范围为0到1
    scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.2)) +
    # 设置标题和轴标签
    labs(title = "C-index across Different Cancers",
         x = "Dataset",
         y = "C-index") +
    # 使用简洁美观的背景主题
    theme_bw()
    # 调整字体、标题居中、X轴标签旋转防重叠
    theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      axis.title.x = element_text(size = 14, margin = margin(t = 10)),
      axis.title.y = element_text(size = 14, margin = margin(r = 10)),
      axis.text.x = element_text(size = 12, angle = 45, hjust = 1, color = "black"),
      axis.text.y = element_text(size = 12, color = "black"),
      panel.grid.major.x = element_blank(), # 隐藏X轴的垂直网格线，使图面更干净
      panel.grid.minor = element_blank()
    )
  
  # 保存图片
  save_path <- file.path(base_dir, "C_index_bar_chart_ggplot.png")
  # ggsave 能够根据尺寸和 dpi 自动输出高质量出版级图片
  ggsave(filename = save_path, plot = p, width = 10, height = 6, dpi = 300)
  
  message(paste("✅ 绘图完成！图片已保存至:", save_path))
}



###############   section5  景观图


# ==============================================================================
# Pan-Cancer Multi-omics Landscape Decoupled by JEPA (Academic Factory Edition)
# 核心特性：全图层临床保留(非干预排序) + 显著差异特征多宫格阵列自动化建档落盘
# ==============================================================================


  library(tidyverse)
  library(data.table)
  library(ComplexHeatmap)
  library(circlize)
  library(RColorBrewer)
  library(ggplot2)
  library(ggpubr)


# ==================== Step 1: 配置参数与绝对路径 ====================
message("\n[1/9] 正在初始化参数与路径配置...")

output_dir <- '/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section4'
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ★ 新建专属子文件夹，分门别类存放海量统计验证子图
val_dir <- file.path(output_dir, 'Statistical_Validations')
dir.create(file.path(val_dir, "Pathways"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(val_dir, "Immune_Cells"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(val_dir, "Mutations"), recursive = TRUE, showWarnings = FALSE)

cancer_list <- c('BRCA', 'COAD') # ⚠️ 测试通过后，替换为完整的 20 种癌症
cancer_list <- c("BRCA" ,"BLCA" ,"UCEC" ,"ESCA" ,"CESC", "COAD" ,"HNSC", "KIRC", "KIRP" ,"LGG" , "LIHC" ,"LUAD" ,"LUSC", "PAAD" ,"PCPG", "PRAD", "SARC" ,"SKCM", "STAD", "THCA")
resolution_map <- c(
  'BLCA'='1.0', 'BRCA'='1.0', 'CESC'='1.6', 'COAD'='1.6', 'ESCA'='1.8', 
  'HNSC'='0.6', 'KIRC'='1.8', 'KIRP'='1.4', 'LGG'='1.4', 'LIHC'='1.2', 
  'LUAD'='0.8', 'LUSC'='1.0', 'PAAD'='1.6', 'PCPG'='1.0', 'PRAD'='1.2', 
  'SARC'='1.4', 'SKCM'='1.2', 'STAD'='1.4', 'THCA'='1.6', 'UCEC'='0.6'
)

clinical_path <- '/home/Disk30T/bjt/cluster_methods/Survival_SupplementalTable_S1_20171025_xena_sp.tsv'
survival_path <- '/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/ALL_OS_dataset.csv'
subtype_path <- '/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/TCGASubtype.20170308.tsv' 
cluster_path_template <- '/home/Disk30T/bjt/cluster_methods/leiden_clustering_results_1/resolution_{res}/{cancer}/align/{cancer_lower}_align_res{res}_labels.csv'
mut_file <- '/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/Org/ALL_TCGA_ALL.somaticmutation_wxs.tsv.gz'
ssgsea_file <- '/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/PanCan33_ssGSEA_1387GeneSets_NonZero_sample_level.txt'
timer_file <- '/home/Disk30T/chengpeng/Multimodal_JEPA/Dataset/TCGA/TCGA_phenotype/infiltration_estimation_for_tcga.csv' 


# ==================== Step 2: 临床融合与【纯净全图层排版】 ====================
message("[2/9] 读取临床数据并执行平滑降噪级联排序...")

# 注意：Gender被统一转为了大写，防止后续色卡字典匹配失败产生空色块
clinical_subset <- read_tsv(clinical_path, show_col_types = FALSE) %>% select(sample_name = sample, Age = age_at_initial_pathologic_diagnosis, Gender = gender, Stage = ajcc_pathologic_tumor_stage) %>% mutate(sample_name = str_sub(sample_name, 1, 12), Gender = toupper(Gender)) %>% distinct(sample_name, .keep_all = TRUE)
survival_subset <- read_csv(survival_path, show_col_types = FALSE) %>% select(sample_name = sample_id, survival_time, survival_event) %>% mutate(sample_name = str_sub(sample_name, 1, 12)) %>% distinct(sample_name, .keep_all = TRUE)
subtype_df <- read_tsv(subtype_path, show_col_types = FALSE) %>% select(sample_name = sampleID, Type = Subtype_Selected) %>% mutate(sample_name = str_sub(sample_name, 1, 12), Type = if_else(is.na(Type) | Type %in% c("NA", ""), NA_character_, Type)) %>% distinct(sample_name, .keep_all = TRUE) 

load_cancer_data <- function(cancer_name) {
  res <- ifelse(cancer_name %in% names(resolution_map), resolution_map[cancer_name], '1.0')
  cluster_p <- str_glue(cluster_path_template, res = res, cancer = cancer_name, cancer_lower = tolower(cancer_name))
  if(!file.exists(cluster_p)) return(NULL) 
  read_csv(cluster_p, show_col_types = FALSE) %>% select(sample_name, cluster) %>% mutate(sample_name = str_sub(sample_name, 1, 12)) %>% distinct(sample_name, .keep_all = TRUE) %>% left_join(clinical_subset, by = "sample_name") %>% mutate(Cancer = cancer_name) %>% left_join(survival_subset, by = "sample_name") %>% left_join(subtype_df, by = "sample_name")
}

combined_sorted <- map_dfr(cancer_list, load_cancer_data) %>%
  mutate(
    Age_Cat = factor(case_when(is.na(Age) ~ NA_character_, Age < 25 ~ "<25", Age < 45 ~ "25-45", Age < 65 ~ "45-65", Age < 85 ~ "65-85", TRUE ~ ">85"), levels = c('<25', '25-45', '45-65', '65-85', '>85')),
    Stage_Num = case_when(grepl("IV|4", Stage, ignore.case=T) ~ 4, grepl("III|3", Stage, ignore.case=T) ~ 3, grepl("II|2", Stage, ignore.case=T) ~ 2, grepl("I|1", Stage, ignore.case=T) ~ 1, TRUE ~ 0),
    Cluster = as.character(cluster), survival_event = as.character(survival_event), Cancer = factor(Cancer, levels = cancer_list)
  ) %>%
  select(sample_name, Cancer, Cluster, Type, Age_Cat, Gender, Stage, Stage_Num, survival_time, survival_event) %>% 
  # 🌟 核心排版逻辑：临床图层 `Age_Cat` 和 `Gender` 已经被 `select` 完美挂载保留。
  # 但在 `arrange` 排序引擎中坚决剥离！这保证了排版底座依然是极致纯净的五维主干瀑布流。
  arrange(Cancer, Cluster, Type, Stage_Num, desc(survival_time)) %>% 
  distinct(sample_name, .keep_all = TRUE)


# ==================== Step 3 & 4: 组学数据读取与 Z-score 对齐 ====================
message("[3/9] & [4/9] 构建与对齐突变、通路、免疫三大高维矩阵...")

mut_raw <- fread(mut_file, select = c("sample", "gene", "effect"))
noise_genes <- c("TTN", "MUC16", "OBSCN", "RYR2", "FLG", "SYNE1", "SYNE2", "LRP1B", "CSMD1", "CSMD3", "USH2A", "MACF1", "DNAH5", "HMCN1")
mut_wide_df <- mut_raw %>% mutate(sample_name = str_sub(sample, 1, 12)) %>% filter(sample_name %in% combined_sorted$sample_name & !gene %in% noise_genes & !grepl("synonymous", effect, ignore.case = TRUE)) %>% mutate(mut_type = case_when(grepl("missense", effect, ignore.case = TRUE) ~ "Missense", grepl("frameshift|stop|nonsense|splice", effect, ignore.case = TRUE) ~ "Truncating", TRUE ~ "Other")) %>% select(sample_name, gene, mut_type) %>% distinct(sample_name, gene, .keep_all = TRUE) %>% pivot_wider(names_from = sample_name, values_from = mut_type, values_fill = "Wild-Type")
mut_mat_full <- as.data.frame(mut_wide_df) %>% column_to_rownames("gene") %>% as.matrix()
missing_samps <- setdiff(combined_sorted$sample_name, colnames(mut_mat_full)); if (length(missing_samps) > 0) { missing_mat <- matrix("Wild-Type", nrow = nrow(mut_mat_full), ncol = length(missing_samps)); colnames(missing_mat) <- missing_samps; mut_mat_full <- cbind(mut_mat_full, missing_mat) }
mut_binary <- (mut_mat_full != "Wild-Type")

ssgsea_raw <- fread(ssgsea_file)
if(grepl("TCGA|sample", colnames(ssgsea_raw)[1], ignore.case = TRUE) || grepl("TCGA", as.character(ssgsea_raw[1,1]))) { colnames(ssgsea_raw)[1] <- "sample_id"; ssgsea_df <- ssgsea_raw %>% mutate(sample_name = gsub("\\.", "-", str_sub(sample_id, 1, 12))) %>% filter(sample_name %in% combined_sorted$sample_name) %>% select(-sample_id) } else { pathways <- ssgsea_raw[[1]]; ssgsea_df <- as.data.frame(ssgsea_raw[, -1, with=FALSE]) %>% mutate(across(everything(), ~ as.character(.))) ; colnames(ssgsea_df) <- gsub("\\.", "-", str_sub(colnames(ssgsea_df), 1, 12)); ssgsea_df <- as.data.frame(t(ssgsea_df)) %>% rownames_to_column("sample_name") %>% filter(sample_name %in% combined_sorted$sample_name); colnames(ssgsea_df)[-1] <- pathways }
ssgsea_mat_full <- ssgsea_df %>% mutate(across(-sample_name, as.numeric)) %>% inner_join(combined_sorted %>% select(sample_name, Cancer), by = "sample_name") %>% group_by(Cancer) %>% mutate(across(where(is.numeric), ~ if(sd(., na.rm=TRUE) == 0) 0 else as.numeric(scale(.)))) %>% ungroup() %>% as.data.frame() %>% select(-Cancer) %>% column_to_rownames("sample_name")

timer_raw <- fread(timer_file)
if(grepl("TCGA|sample", colnames(timer_raw)[1], ignore.case = TRUE) || grepl("TCGA", as.character(timer_raw[1,1]))) { colnames(timer_raw)[1] <- "sample_id"; timer_df <- timer_raw %>% mutate(sample_name = gsub("\\.", "-", str_sub(sample_id, 1, 12))) %>% distinct(sample_name, .keep_all = TRUE) %>% filter(sample_name %in% combined_sorted$sample_name) %>% select(-sample_id) } else { cells <- timer_raw[[1]]; timer_mat <- t(as.matrix(timer_raw[, -1, with=FALSE])); timer_df <- as.data.frame(timer_mat); timer_df$sample_name <- gsub("\\.", "-", str_sub(rownames(timer_df), 1, 12)); rownames(timer_df) <- NULL; timer_df <- timer_df %>% distinct(sample_name, .keep_all = TRUE) %>% filter(sample_name %in% combined_sorted$sample_name) %>% select(sample_name, everything()); colnames(timer_df)[-1] <- cells }
timer_z_df <- timer_df %>% mutate(across(-sample_name, as.numeric)) %>% inner_join(combined_sorted %>% select(sample_name, Cancer), by = "sample_name") %>% group_by(Cancer) %>% mutate(across(where(is.numeric), ~ if(sd(., na.rm=TRUE) == 0) 0 else as.numeric(scale(.)))) %>% ungroup() %>% as.data.frame()
timer_mat_full <- timer_z_df %>% select(-Cancer) %>% column_to_rownames("sample_name")
missing_timer <- setdiff(combined_sorted$sample_name, rownames(timer_mat_full)); if (length(missing_timer) > 0) { na_mat <- matrix(NA, nrow = length(missing_timer), ncol = ncol(timer_mat_full)); rownames(na_mat) <- missing_timer; colnames(na_mat) <- colnames(timer_mat_full); timer_mat_full <- rbind(timer_mat_full, na_mat) }

common_samps <- intersect(combined_sorted$sample_name, intersect(colnames(mut_mat_full), rownames(ssgsea_mat_full)))
combined_sorted <- combined_sorted %>% filter(sample_name %in% common_samps) 
mut_mat_full <- mut_mat_full[, combined_sorted$sample_name, drop = FALSE]          
mut_binary <- mut_binary[, combined_sorted$sample_name, drop = FALSE] 
canc_path_full <- as.matrix(ssgsea_mat_full[combined_sorted$sample_name, , drop = FALSE])
canc_timer_full <- as.matrix(timer_mat_full[combined_sorted$sample_name, , drop = FALSE])


# ==================== Step 5: 严谨统计检验 (Wilcoxon/Fisher + FDR校正) ====================
message("[5/9] 启用 Wilcoxon与Fisher 统计推断引擎，输出带 FDR P值 的高定特征表，并为可视化抓取显著特征...")

selected_muts <- character(); selected_paths <- character(); selected_timers <- character()
feature_export_list <- list() 
cancer_plot_features <- list() 

clean_name_fn <- function(x) gsub("_([A-Za-z0-9]+)$", " (\\1)", gsub("_", " ", x))
format_p <- function(feats, pvals) { if(length(feats)==0) return("None"); paste0(feats, " (FDR=", formatC(pvals[feats], format="e", digits=2), ")", collapse = " | ") }

for (canc in unique(combined_sorted$Cancer)) {
  print(canc)
  samps_idx <- which(combined_sorted$Cancer == canc)
  if (length(samps_idx) < 10) next 
  
  canc_clusters <- combined_sorted$Cluster[samps_idx]
  canc_mut <- mut_binary[, samps_idx, drop = FALSE]
  canc_path <- canc_path_full[samps_idx, , drop = FALSE]
  canc_timer <- canc_timer_full[samps_idx, , drop = FALSE]
  
  canc_mut <- canc_mut[names(which(rowMeans(canc_mut, na.rm=TRUE) >= 0.02)), , drop = FALSE] 
  if(nrow(canc_mut) == 0) next
  
  # 🌟 升级：初始化专属显著特征及对应的 FDR P值收集器
  sig_mut_canc <- character(); sig_path_canc <- character(); sig_timer_canc <- character()
  fdr_mut_canc <- numeric(); fdr_path_canc <- numeric(); fdr_timer_canc <- numeric()
  
  for (clu in unique(canc_clusters)) {
    in_clu <- (canc_clusters == clu)
    if (sum(in_clu) < 3 || sum(!in_clu) < 3) next
    
    # [1] Pathway
    p_vals_p <- apply(canc_path, 2, function(x) { if(length(unique(x[!is.na(x)]))<2) return(1); tryCatch(wilcox.test(x[in_clu], x[!in_clu], alternative="greater", exact=FALSE)$p.value, error=function(e) 1) })
    fdr_p <- p.adjust(p_vals_p, method="BH")
    diff_p <- colMeans(canc_path[in_clu,,drop=F], na.rm=T) - colMeans(canc_path[!in_clu,,drop=F], na.rm=T)
    valid_p <- which(fdr_p < 0.05 & diff_p > 0)
    str_paths <- "None"
    if(length(valid_p) > 0) { 
      ord_p <- valid_p[order(fdr_p[valid_p])]
      sel_p <- names(ord_p)[1:min(8, length(ord_p))]
      selected_paths <- c(selected_paths, names(ord_p)[1:min(3, length(ord_p))])
      sig_path_canc <- c(sig_path_canc, sel_p)
      fdr_path_canc <- c(fdr_path_canc, setNames(fdr_p[sel_p], sel_p)) # 记录具体 P值
      str_paths <- format_p(names(ord_p)[1:min(10, length(ord_p))], fdr_p) 
    }
    
    # [2] Immune
    p_vals_t <- apply(canc_timer, 2, function(x) { if(length(unique(x[!is.na(x)]))<2) return(1); tryCatch(wilcox.test(x[in_clu], x[!in_clu], alternative="greater", exact=FALSE)$p.value, error=function(e) 1) })
    fdr_t <- p.adjust(p_vals_t, method="BH")
    diff_t <- colMeans(canc_timer[in_clu,,drop=F], na.rm=T) - colMeans(canc_timer[!in_clu,,drop=F], na.rm=T)
    valid_t <- which(fdr_t < 0.05 & diff_t > 0)
    str_timers <- "None"
    if(length(valid_t) > 0) { 
      ord_t <- valid_t[order(fdr_t[valid_t])]
      sel_t <- names(ord_t)[1:min(8, length(ord_t))]
      selected_timers <- c(selected_timers, names(ord_t)[1:min(3, length(ord_t))])
      sig_timer_canc <- c(sig_timer_canc, sel_t)
      fdr_timer_canc <- c(fdr_timer_canc, setNames(fdr_t[sel_t], sel_t)) # 记录具体 P值
      fdr_t_clean <- fdr_t; names(fdr_t_clean) <- clean_name_fn(names(fdr_t))
      str_timers <- format_p(clean_name_fn(names(ord_t))[1:min(10, length(ord_t))], fdr_t_clean) 
    }
    
    # [3] Mutation
    freq_in <- rowMeans(canc_mut[, in_clu, drop=F], na.rm=T); freq_out <- rowMeans(canc_mut[, !in_clu, drop=F], na.rm=T)
    diff_m <- freq_in - freq_out; fc_m <- (freq_in + 1e-5) / (freq_out + 1e-5)
    p_vals_m <- apply(canc_mut, 1, function(x) {
      mut_in <- sum(x[in_clu], na.rm=TRUE); nomut_in <- sum(!x[in_clu], na.rm=TRUE); mut_out <- sum(x[!in_clu], na.rm=TRUE); nomut_out <- sum(!x[!in_clu], na.rm=TRUE)
      cont_mat <- matrix(c(mut_in, nomut_in, mut_out, nomut_out), nrow = 2, byrow = FALSE)
      tryCatch(fisher.test(cont_mat, alternative="greater")$p.value, error=function(e) 1)
    })
    fdr_m <- p.adjust(p_vals_m, method="BH")
    valid_m <- which(fdr_m < 0.05 & diff_m > 0.05 & fc_m > 1.2)
    str_muts <- "None"
    if(length(valid_m) > 0) { 
      ord_m <- valid_m[order(fdr_m[valid_m])]
      sel_m <- names(ord_m)[1:min(8, length(ord_m))]
      selected_muts <- c(selected_muts, names(ord_m)[1:min(5, length(ord_m))])
      sig_mut_canc <- c(sig_mut_canc, sel_m)
      fdr_mut_canc <- c(fdr_mut_canc, setNames(fdr_m[sel_m], sel_m)) # 记录具体 P值
      str_muts <- format_p(names(ord_m)[1:min(10, length(ord_m))], fdr_m) 
    }
    
    feature_export_list[[length(feature_export_list) + 1]] <- data.frame(Cancer_Type = canc, Cluster_ID = clu, Enriched_Mutations = str_muts, Enriched_Pathways = str_paths, Enriched_Immune_Cells = str_timers, stringsAsFactors = FALSE)
  }
  
  # 🌟 辅助函数：针对跨亚组重复出现的特征，提取最小（最显著）的 FDR 数值
  get_min_fdr_map <- function(fdr_vec) {
    if(length(fdr_vec) == 0) return(numeric(0))
    tapply(fdr_vec, names(fdr_vec), min)
  }
  
  # 🌟 保存特征的同时，挂载精准的 FDR 数值映射字典
  cancer_plot_features[[canc]] <- list(
    muts = unique(sig_mut_canc),
    paths = unique(sig_path_canc),
    timers = unique(sig_timer_canc),
    fdr_map = c(get_min_fdr_map(fdr_mut_canc), get_min_fdr_map(fdr_path_canc), get_min_fdr_map(fdr_timer_canc))
  )
}

export_csv_path <- file.path(output_dir, "Cluster_Biomarkers_Statistical_FDR.csv")
write_csv(bind_rows(feature_export_list), export_csv_path)
# message("[5/9] 启用 Wilcoxon与Fisher 统计推断引擎，输出带 FDR P值 的高定特征表，并为可视化抓取显著特征...")
# 
# selected_muts <- character(); selected_paths <- character(); selected_timers <- character()
# feature_export_list <- list() 
# cancer_plot_features <- list() # 🌟 新增：用于为 Step 9 精准记录每个癌症独有的 FDR 显著特征池
# 
# clean_name_fn <- function(x) gsub("_([A-Za-z0-9]+)$", " (\\1)", gsub("_", " ", x))
# format_p <- function(feats, pvals) { if(length(feats)==0) return("None"); paste0(feats, " (FDR=", formatC(pvals[feats], format="e", digits=2), ")", collapse = " | ") }
# 
# for (canc in unique(combined_sorted$Cancer)) {
#   print(canc)
#   samps_idx <- which(combined_sorted$Cancer == canc)
#   if (length(samps_idx) < 10) next 
#   
#   canc_clusters <- combined_sorted$Cluster[samps_idx]
#   canc_mut <- mut_binary[, samps_idx, drop = FALSE]
#   canc_path <- canc_path_full[samps_idx, , drop = FALSE]
#   canc_timer <- canc_timer_full[samps_idx, , drop = FALSE]
#   
#   canc_mut <- canc_mut[names(which(rowMeans(canc_mut, na.rm=TRUE) >= 0.02)), , drop = FALSE] 
#   if(nrow(canc_mut) == 0) next
#   
#   # 🌟 为该癌症初始化专属的显著特征收集器 (用于 Step 9 出图)
#   sig_mut_canc <- character(); sig_path_canc <- character(); sig_timer_canc <- character()
#   
#   for (clu in unique(canc_clusters)) {
#     in_clu <- (canc_clusters == clu)
#     if (sum(in_clu) < 3 || sum(!in_clu) < 3) next
#     
#     # [1] Pathway
#     p_vals_p <- apply(canc_path, 2, function(x) { if(length(unique(x[!is.na(x)]))<2) return(1); tryCatch(wilcox.test(x[in_clu], x[!in_clu], alternative="greater", exact=FALSE)$p.value, error=function(e) 1) })
#     fdr_p <- p.adjust(p_vals_p, method="BH")
#     diff_p <- colMeans(canc_path[in_clu,,drop=F], na.rm=T) - colMeans(canc_path[!in_clu,,drop=F], na.rm=T)
#     valid_p <- which(fdr_p < 0.05 & diff_p > 0)
#     str_paths <- "None"
#     if(length(valid_p) > 0) { 
#       ord_p <- valid_p[order(fdr_p[valid_p])]
#       selected_paths <- c(selected_paths, names(ord_p)[1:min(3, length(ord_p))])
#       sig_path_canc <- c(sig_path_canc, names(ord_p)[1:min(8, length(ord_p))]) # 🌟 收集该亚组 Top 8 显著通路
#       str_paths <- format_p(names(ord_p)[1:min(10, length(ord_p))], fdr_p) 
#     }
#     
#     # [2] Immune
#     p_vals_t <- apply(canc_timer, 2, function(x) { if(length(unique(x[!is.na(x)]))<2) return(1); tryCatch(wilcox.test(x[in_clu], x[!in_clu], alternative="greater", exact=FALSE)$p.value, error=function(e) 1) })
#     fdr_t <- p.adjust(p_vals_t, method="BH")
#     diff_t <- colMeans(canc_timer[in_clu,,drop=F], na.rm=T) - colMeans(canc_timer[!in_clu,,drop=F], na.rm=T)
#     valid_t <- which(fdr_t < 0.05 & diff_t > 0)
#     str_timers <- "None"
#     if(length(valid_t) > 0) { 
#       ord_t <- valid_t[order(fdr_t[valid_t])]
#       selected_timers <- c(selected_timers, names(ord_t)[1:min(3, length(ord_t))])
#       sig_timer_canc <- c(sig_timer_canc, names(ord_t)[1:min(8, length(ord_t))]) # 🌟 收集该亚组 Top 8 显著免疫细胞
#       fdr_t_clean <- fdr_t; names(fdr_t_clean) <- clean_name_fn(names(fdr_t))
#       str_timers <- format_p(clean_name_fn(names(ord_t))[1:min(10, length(ord_t))], fdr_t_clean) 
#     }
#     
#     # [3] Mutation (绝对安全的 2x2 列联表手工生成计算)
#     freq_in <- rowMeans(canc_mut[, in_clu, drop=F], na.rm=T); freq_out <- rowMeans(canc_mut[, !in_clu, drop=F], na.rm=T)
#     diff_m <- freq_in - freq_out; fc_m <- (freq_in + 1e-5) / (freq_out + 1e-5)
#     p_vals_m <- apply(canc_mut, 1, function(x) {
#       mut_in <- sum(x[in_clu], na.rm=TRUE); nomut_in <- sum(!x[in_clu], na.rm=TRUE); mut_out <- sum(x[!in_clu], na.rm=TRUE); nomut_out <- sum(!x[!in_clu], na.rm=TRUE)
#       cont_mat <- matrix(c(mut_in, nomut_in, mut_out, nomut_out), nrow = 2, byrow = FALSE)
#       tryCatch(fisher.test(cont_mat, alternative="greater")$p.value, error=function(e) 1)
#     })
#     fdr_m <- p.adjust(p_vals_m, method="BH")
#     valid_m <- which(fdr_m < 0.05 & diff_m > 0.05 & fc_m > 1.2)
#     str_muts <- "None"
#     if(length(valid_m) > 0) { 
#       ord_m <- valid_m[order(fdr_m[valid_m])]
#       selected_muts <- c(selected_muts, names(ord_m)[1:min(5, length(ord_m))])
#       sig_mut_canc <- c(sig_mut_canc, names(ord_m)[1:min(8, length(ord_m))]) # 🌟 收集该亚组 Top 8 显著突变
#       str_muts <- format_p(names(ord_m)[1:min(10, length(ord_m))], fdr_m) 
#     }
#     
#     feature_export_list[[length(feature_export_list) + 1]] <- data.frame(Cancer_Type = canc, Cluster_ID = clu, Enriched_Mutations = str_muts, Enriched_Pathways = str_paths, Enriched_Immune_Cells = str_timers, stringsAsFactors = FALSE)
#   }
#   
#   # 🌟 保存该癌症专属的去重显著特征池
#   cancer_plot_features[[canc]] <- list(
#     muts = unique(sig_mut_canc),
#     paths = unique(sig_path_canc),
#     timers = unique(sig_timer_canc)
#   )
# }
# 
# export_csv_path <- file.path(output_dir, "1Cluster_Biomarkers_Statistical_FDR.csv")
# write_csv(bind_rows(feature_export_list), export_csv_path)






# --- 生成景观大图矩阵 ---
num_cancers <- length(unique(combined_sorted$Cancer))
max_muts <- min(80, max(25, num_cancers * 4)); max_paths <- min(60, max(20, num_cancers * 3)); max_timers <- min(50, max(20, num_cancers * 3))

top_paths <- names(sort(table(selected_paths), decreasing = TRUE))
top_paths <- if(length(top_paths)>0) top_paths[1:min(max_paths, length(top_paths))] else colnames(canc_path_full)[1:max_paths]
top_muts <- names(sort(table(selected_muts), decreasing = TRUE))
if (length(top_muts) < max_muts) { top_muts <- unique(c(top_muts, names(sort(rowSums(mut_binary, na.rm=TRUE), decreasing = TRUE))))[1:min(max_muts, length(unique(c(top_muts, names(sort(rowSums(mut_binary, na.rm=TRUE), decreasing = TRUE))))))] } else { top_muts <- top_muts[1:max_muts] }
top_timers <- names(sort(table(selected_timers), decreasing = TRUE))
top_timers <- if(length(top_timers)>0) top_timers[1:min(max_timers, length(top_timers))] else names(sort(apply(canc_timer_full, 2, var, na.rm=TRUE), decreasing=TRUE)[1:min(max_timers, ncol(canc_timer_full))])

plot_mut <- mut_mat_full[top_muts, combined_sorted$sample_name, drop = FALSE]
plot_path <- t(as.matrix(canc_path_full[, top_paths, drop = FALSE]))
plot_timer <- t(as.matrix(canc_timer_full[, top_timers, drop = FALSE]))
rownames(plot_timer) <- clean_name_fn(rownames(plot_timer))


# ==================== Step 6-8: 绘制底置排版级联排序泛癌大图 ====================
message("[6-8/9] 渲染级联排序与高对比度配色的宏观景观图...")

cancer_colors <- if(length(cancer_list) <= 2) c('#d73027', '#2166ac') else colorRampPalette(brewer.pal(8, "Set1"))(length(cancer_list)); names(cancer_colors) <- cancer_list
unique_stages <- sort(unique(na.omit(combined_sorted$Stage))); stage_colors <- setNames(colorRampPalette(brewer.pal(9, "YlGnBu"))(length(unique_stages)), unique_stages)
survival_time_fun <- colorRamp2(c(0, max(combined_sorted$survival_time, na.rm = TRUE)/2, max(combined_sorted$survival_time, na.rm = TRUE)), c('#e0ecf4', '#6a51a3', '#4d004b'))
survival_event_colors <- c('0' = '#f0f0f0', '1' = '#000000')
age_colors <- setNames(c('#fff7bc', '#fec44f', '#fe9929', '#ec7014', '#cc4c02'), c('<25', '25-45', '45-65', '65-85', '>85'))
gender_colors <- c('MALE' = '#7b3294', 'FEMALE' = '#008837')

mut_colors <- c("Missense" = "#008000", "Truncating" = "#FF0000", "Other" = "#0000FF", "Wild-Type" = "#F0F0F0")
path_colors <- colorRamp2(c(-2, 0, 2), c("navy", "white", "firebrick3"))
timer_colors <- colorRamp2(c(-2, 0, 2), c("#5e3c99", "white", "#e66101")) 

raw_master_pool <- unique(c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666", "#1F78B4", "#33A02C", "#E31A1C", "#6A3D9A", "#B15928", "#FDBF6F", "#CAB2D6", "#B2DF8A", "#A6CEE3", "#FB9A99", "#7FC97F", "#BEAED4", "#FDC086", "#386CB0", "#F0027F", "#BF5B17", "#8DD3C7", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9"))
master_contrast_pool <- raw_master_pool[!raw_master_pool %in% c("#FFFF33", "#D9D9D9", "#F2F2F2", "#B3B3B3", "#FFED6F", "#FFFF99", "#E5D8BD", "#FDDAEC", "#CCEBC5")]
cluster_pool <- rep(master_contrast_pool, 10); subtype_pool <- rep(rev(master_contrast_pool), 10)     

combined_sorted <- combined_sorted %>% mutate(Global_Cluster = paste0(Cancer, "_C", Cluster), Global_Subtype = if_else(is.na(Type), NA_character_, paste0(Cancer, "_", Type)))
custom_legends <- list(); global_cluster_colors <- c(); global_subtype_colors <- c(); has_subtype <- any(!is.na(combined_sorted$Type))

for (cc in cancer_list) {
  sub_df <- combined_sorted %>% filter(Cancer == cc); if(nrow(sub_df) == 0) next
  sub_clusters <- sort(unique(sub_df$Cluster)); cc_c_colors <- cluster_pool[1:length(sub_clusters)]; names(cc_c_colors) <- paste0(cc, "_C", sub_clusters); global_cluster_colors <- c(global_cluster_colors, cc_c_colors); names(cc_c_colors) <- sub_clusters; custom_legends <- append(custom_legends, list(Legend(title = paste0("Cluster (", cc, ")"), labels = sub_clusters, legend_gp = gpar(fill = cc_c_colors), title_gp = gpar(fontface = "bold", fontsize = 10), nrow = 1)))
  if(has_subtype) { sub_types <- sort(unique(na.omit(sub_df$Type))); if (length(sub_types) > 0) { cc_t_colors <- subtype_pool[1:length(sub_types)]; names(cc_t_colors) <- paste0(cc, "_", sub_types); global_subtype_colors <- c(global_subtype_colors, cc_t_colors); names(cc_t_colors) <- sub_types; custom_legends <- append(custom_legends, list(Legend(title = paste0("Subtype (", cc, ")"), labels = sub_types, legend_gp = gpar(fill = cc_t_colors), title_gp = gpar(fontface = "bold", fontsize = 10), nrow = 2))) } }
}
custom_legends <- append(custom_legends, list(Legend(title = "Mutation Type", labels = c("Missense", "Truncating", "Other", "Wild-Type"), legend_gp = gpar(fill = mut_colors), title_gp = gpar(fontface = "bold"), nrow = 1), Legend(title = "Pathway (Z-Score)", col_fun = path_colors, title_gp = gpar(fontface = "bold"), direction = "horizontal", legend_width = unit(4, "cm")), Legend(title = "Immune Infiltration (Z-Score)", col_fun = timer_colors, title_gp = gpar(fontface = "bold"), direction = "horizontal", legend_width = unit(4, "cm"))))

anno_leg_params <- list(Cancer = list(nrow = ceiling(num_cancers/8)), Age = list(nrow=1), Gender = list(nrow=1), Stage = list(nrow = 2), `OS event` = list(nrow=1), `OS time` = list(direction = "horizontal", title_position = "lefttop", legend_width = unit(3, "cm")))

# 🌟 全图层恢复：Gender 和 Age_Cat 回归注释轨道，完美呈现纯天然学术散布感
anno_df <- combined_sorted %>% select(Cancer, Cluster = Global_Cluster, Age = Age_Cat, Gender, Stage, `OS time` = survival_time, `OS event` = survival_event) %>% as.data.frame()
anno_cols <- list(Cancer = cancer_colors, Cluster = global_cluster_colors, Age = age_colors, Gender = gender_colors, Stage = stage_colors, `OS time` = survival_time_fun, `OS event` = survival_event_colors)
show_leg_list <- c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE)

if(has_subtype) { anno_df <- anno_df %>% mutate(Subtype = combined_sorted$Global_Subtype) %>% relocate(Subtype, .after = Cluster); anno_cols$Subtype <- global_subtype_colors; show_leg_list <- append(show_leg_list, FALSE, after = 2) }

ha <- HeatmapAnnotation(df = anno_df, col = anno_cols, show_legend = show_leg_list, na_col = "#d3d3d3", annotation_name_side = "left", annotation_name_gp = gpar(fontsize = 11, fontface = "bold"), simple_anno_size = unit(0.4, "cm"), annotation_legend_param = anno_leg_params)

plot_mut[is.na(plot_mut) | plot_mut == "Wild-type" | plot_mut == ""] <- "Wild-Type"; plot_mut_aligned <- plot_mut[order(rowSums(plot_mut != "Wild-Type"), decreasing = TRUE), , drop = FALSE]
split_factor <- combined_sorted$Cancer; column_titles <- paste0(levels(split_factor), "\n(n=", as.numeric(table(split_factor)[levels(split_factor)]), ")")

border_col <- if(ncol(plot_mut_aligned) > 1500) NA else "white"
ht_mut <- Heatmap(plot_mut_aligned, name = "Mutation", top_annotation = ha, col = mut_colors, column_split = split_factor, cluster_columns = FALSE, cluster_rows = FALSE, show_column_names = FALSE, row_names_side = "left", row_names_gp = gpar(fontsize = 8, fontface = "italic"), row_names_max_width = max_text_width(rownames(plot_mut_aligned), gp = gpar(fontsize = 8)), show_heatmap_legend = FALSE, rect_gp = gpar(col = border_col, lwd = 0.5), column_title = column_titles, column_title_gp = gpar(fontsize = 12, fontface = "bold", fill = cancer_colors[levels(split_factor)], col = "white", border = "black"), height = unit(nrow(plot_mut_aligned) * 0.35, "cm"))
ht_path <- Heatmap(plot_path, name = "Pathway", col = path_colors, column_split = split_factor, cluster_columns = FALSE, cluster_rows = TRUE, show_column_names = FALSE, row_names_side = "left", row_names_gp = gpar(fontsize = 8), row_names_max_width = max_text_width(rownames(plot_path), gp = gpar(fontsize = 8)), show_heatmap_legend = FALSE, height = unit(nrow(plot_path) * 0.35, "cm"))
ht_timer <- Heatmap(plot_timer, name = "Immune", col = timer_colors, column_split = split_factor, cluster_columns = FALSE, cluster_rows = TRUE, show_column_names = FALSE, row_names_side = "left", row_names_gp = gpar(fontsize = 8), row_names_max_width = max_text_width(rownames(plot_timer), gp = gpar(fontsize = 8)), show_heatmap_legend = FALSE, na_col = "#e0e0e0", height = unit(nrow(plot_timer) * 0.35, "cm"))

ht_final <- ht_mut %v% ht_path %v% ht_timer
output_pdf_path <- file.path(output_dir, 'Figure4_Multiomics_Landscape.pdf')
pdf(output_pdf_path, width = min(35, max(12, nrow(combined_sorted) * 0.0025 + 8)), height = max(20, 10 + (nrow(plot_mut_aligned) + nrow(plot_path) + nrow(plot_timer)) * 0.15))
draw(ht_final, annotation_legend_side = "bottom", heatmap_legend_side = "bottom", annotation_legend_list = custom_legends, merge_legend = TRUE, padding = unit(c(10, 5, 20, 5), "mm"), column_title = "Pan-Cancer Multi-omics Landscape Decoupled by JEPA", column_title_gp = gpar(fontsize = 22, fontface = "bold"))
dev.off()


# ==================== Step 9: 工业级自动化流水线 (遍历出图+智能靶向落盘) ====================
# ==================== Step 9: 工业级自动化流水线 (按癌症出图 + 靶向落盘 + 统计表导出) ====================
message("\n[9/9] 工业流水线引擎启动：基于 Step 5 筛选的 FDR 显著特征，按癌症绘制数值化大图 PDF...")

violin_df <- combined_sorted %>% select(sample_name, Cancer, Cluster) %>% drop_na(Cancer, Cluster)
cls_levels <- sort(unique(violin_df$Cluster))
cls_colors <- setNames(cluster_pool[1:length(cls_levels)], cls_levels)

val_dir_cancer <- file.path(output_dir, 'Cancer_Specific_Validations')
dir.create(val_dir_cancer, recursive = TRUE, showWarnings = FALSE)

# 🌟 新增：精准格式化 FDR P值并附加星号的格式化函数
format_fdr_label <- function(p) {
  if (is.null(p) || is.na(p)) return("FDR Sig.")
  stars <- ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "")))
  sprintf("FDR = %.2e %s", p, stars)
}

for (canc in unique(violin_df$Cancer)) {
  sub_d <- violin_df %>% filter(Cancer == canc)
  if (length(unique(sub_d$Cluster)) < 2) next
  
  sig_muts_canc <- if(!is.null(cancer_plot_features[[canc]])) cancer_plot_features[[canc]]$muts else character()
  sig_paths_canc <- if(!is.null(cancer_plot_features[[canc]])) cancer_plot_features[[canc]]$paths else character()
  sig_timers_canc <- if(!is.null(cancer_plot_features[[canc]])) cancer_plot_features[[canc]]$timers else character()
  fdr_map <- if(!is.null(cancer_plot_features[[canc]])) cancer_plot_features[[canc]]$fdr_map else numeric()
  
  if (length(sig_muts_canc) == 0 && length(sig_paths_canc) == 0 && length(sig_timers_canc) == 0) {
    message(sprintf("  -> 忽略 %s: 未发现满足严格 FDR < 0.05 阈值的显著特征。", canc))
    next
  }
  
  message(sprintf("  -> 正在渲染 %s 专属特征数值化大图 (Muts=%d, Paths=%d, Imm=%d)...", 
                  canc, length(sig_muts_canc), length(sig_paths_canc), length(sig_timers_canc)))
  
  plots_mut <- list(); plots_path <- list(); plots_imm <- list()
  
  # --- 1. 突变 (Mutations) ---
  for (feat in sig_muts_canc) {
    if (!feat %in% rownames(mut_binary)) next
    feat_data <- mut_binary[feat, ]
    sub_d_feat <- sub_d %>% mutate(Val_Score = as.numeric(feat_data[sample_name])) %>% drop_na(Val_Score)
    if (sum(sub_d_feat$Val_Score, na.rm=TRUE) == 0) next
    
    clu_stats <- sub_d_feat %>% group_by(Cluster) %>% summarise(v = mean(Val_Score, na.rm=TRUE), .groups="drop") %>% arrange(desc(v))
    high_clu <- clu_stats$Cluster[1]
    feat_clean <- gsub("_", " ", feat)
    feat_wrapped <- stringr::str_wrap(feat_clean, width = 15)
    
    # 🌟 提取具体 FDR 数值嵌入标题
    fdr_label <- format_fdr_label(fdr_map[feat])
    p_title <- sprintf("%s\nEnriched in C%s (%s)", feat_wrapped, high_clu, fdr_label)
    
    # 🌟 柱状图上方追加具体的突变百分比数值 (geom_text)
    p <- ggbarplot(clu_stats %>% mutate(v_pct = v*100), x="Cluster", y="v_pct", fill="Cluster") + 
      geom_text(aes(label = sprintf("%.1f%%", v_pct)), vjust = -0.3, size = 3, fontface = "bold") +
      scale_fill_manual(values=cls_colors) + theme_bw() + 
      scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
      labs(title=p_title, x="", y="Freq (%)") + 
      theme(legend.position="none", plot.title=element_text(size=9, face="bold", hjust=0.5, color="#E41A1C"))
    plots_mut[[feat]] <- p
  }
  
  # --- 2. 通路 (Pathways) ---
  for (feat in sig_paths_canc) {
    if (!feat %in% colnames(canc_path_full)) next
    feat_data <- canc_path_full[, feat]
    sub_d_feat <- sub_d %>% mutate(Val_Score = as.numeric(feat_data[sample_name])) %>% drop_na(Val_Score)
    if (nrow(sub_d_feat) == 0) next
    
    clu_stats <- sub_d_feat %>% group_by(Cluster) %>% summarise(v = mean(Val_Score, na.rm=TRUE), .groups="drop") %>% arrange(desc(v))
    high_clu <- clu_stats$Cluster[1]
    feat_clean <- gsub("_", " ", feat)
    feat_wrapped <- stringr::str_wrap(feat_clean, width = 15)
    
    # 🌟 提取具体 FDR 数值嵌入标题
    fdr_label <- format_fdr_label(fdr_map[feat])
    p_title <- sprintf("%s\nEnriched in C%s (%s)", feat_wrapped, high_clu, fdr_label)
    
    # 🌟 图内追加 Kruskal-Wallis 全局显著性 P 值标签
    p <- ggviolin(sub_d_feat, x="Cluster", y="Val_Score", fill="Cluster", add="boxplot", add.params=list(fill="white", width=0.1), trim=FALSE) + 
      stat_compare_means(method = "kruskal.test", label = "p.format", label.x.npc = "center", label.y.npc = "top", size = 3) +
      scale_fill_manual(values=cls_colors) + theme_bw() + 
      scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
      labs(title=p_title, x="", y="Z-Score") + 
      theme(legend.position="none", plot.title=element_text(size=9, face="bold", hjust=0.5, color="#377EB8"))
    plots_path[[feat]] <- p
  }
  
  # --- 3. 免疫细胞 (Immune Cells) ---
  for (orig_feat in sig_timers_canc) {
    if (!orig_feat %in% colnames(canc_timer_full)) next
    feat_data <- canc_timer_full[, orig_feat]
    sub_d_feat <- sub_d %>% mutate(Val_Score = as.numeric(feat_data[sample_name])) %>% drop_na(Val_Score)
    if (nrow(sub_d_feat) == 0) next
    
    clu_stats <- sub_d_feat %>% group_by(Cluster) %>% summarise(v = mean(Val_Score, na.rm=TRUE), .groups="drop") %>% arrange(desc(v))
    high_clu <- clu_stats$Cluster[1]
    feat_clean <- clean_name_fn(orig_feat)
    feat_clean <- gsub("_", " ", feat_clean)
    feat_wrapped <- stringr::str_wrap(feat_clean, width = 15)
    
    # 🌟 提取具体 FDR 数值嵌入标题 (通过原始名从字典获取)
    fdr_label <- format_fdr_label(fdr_map[orig_feat])
    p_title <- sprintf("%s\nEnriched in C%s (%s)", feat_wrapped, high_clu, fdr_label)
    
    # 🌟 图内追加 Kruskal-Wallis 全局显著性 P 值标签
    p <- ggviolin(sub_d_feat, x="Cluster", y="Val_Score", fill="Cluster", add="boxplot", add.params=list(fill="white", width=0.1), trim=FALSE) + 
      stat_compare_means(method = "kruskal.test", label = "p.format", label.x.npc = "center", label.y.npc = "top", size = 3) +
      scale_fill_manual(values=cls_colors) + theme_bw() + 
      scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
      labs(title=p_title, x="", y="Z-Score") + 
      theme(legend.position="none", plot.title=element_text(size=9, face="bold", hjust=0.5, color="#4DAF4A"))
    plots_imm[[feat_clean]] <- p
  }
  
  # === 整合并输出单癌症大 PDF ===
  if (length(plots_mut) > 0 || length(plots_path) > 0 || length(plots_imm) > 0) {
    pdf_path <- file.path(val_dir_cancer, paste0(canc, "_Significant_Features.pdf"))
    
    calc_rows <- function(plist) { if(length(plist)==0) 0 else ceiling(length(plist)/6) }
    total_rows <- calc_rows(plots_mut) + calc_rows(plots_path) + calc_rows(plots_imm)
    
    pdf(pdf_path, width = 20, height = max(6, total_rows * 3.5))
    
    arrange_group <- function(plist, title) {
      if(length(plist) == 0) return(NULL)
      ggarrange(plotlist = plist, ncol = min(6, length(plist)), nrow = calc_rows(plist)) %>% 
        annotate_figure(top = text_grob(title, face="bold", size=16, color="black", just="left", x=0.02))
    }
    
    final_arrangements <- list()
    if(length(plots_mut) > 0) final_arrangements[[length(final_arrangements)+1]] <- arrange_group(plots_mut, "Top Significant Mutations (FDR < 0.05)")
    if(length(plots_path) > 0) final_arrangements[[length(final_arrangements)+1]] <- arrange_group(plots_path, "Top Significant Pathways (FDR < 0.05)")
    if(length(plots_imm) > 0) final_arrangements[[length(final_arrangements)+1]] <- arrange_group(plots_imm, "Top Significant Immune Cells (FDR < 0.05)")
    
    print(ggarrange(plotlist = final_arrangements, ncol = 1, heights = c(calc_rows(plots_mut), calc_rows(plots_path), calc_rows(plots_imm))))
    dev.off()
  }
}

message(paste("\n🎉 全栈流水线大满贯完美收官!\n 1. 景观大图与FDR总表已归档\n 2. 基于严格 FDR < 0.05 筛选的具体数值化特征 PDF 阵列已落盘至:", val_dir_cancer))
  
# message("\n[9/9] 工业流水线引擎启动：按癌症类型生成分类特征阵列，并汇总全域显著特征详情表...")
# 
# # 基础数据准备
# violin_df <- combined_sorted %>% select(sample_name, Cancer, Cluster) %>% drop_na(Cancer, Cluster)
# cls_levels <- sort(unique(violin_df$Cluster))
# cls_colors <- setNames(cluster_pool[1:length(cls_levels)], cls_levels)
# safe_filename <- function(x) gsub("[^A-Za-z0-9_.-]", "_", x) 
# 
# # 用于存储统计表格的列表
# detailed_stats_list <- list()
# val_dir_cancer <- file.path(output_dir, 'Cancer_Specific_Validations')
# dir.create(val_dir_cancer, recursive = TRUE, showWarnings = FALSE)
# 
# # 核心：按癌症遍历生成专属 PDF
# for (canc in unique(violin_df$Cancer)) {
#   sub_d <- violin_df %>% filter(Cancer == canc)
#   if (length(unique(sub_d$Cluster)) < 2) next
#   
#   message(sprintf("  -> 正在处理 %s (N=%d)...", canc, nrow(sub_d)))
#   
#   # 存储三类图的列表
#   plots_mut <- list()
#   plots_path <- list()
#   plots_imm <- list()
#   
#   # --- 1. 突变 (Mutations) ---
#   for (feat in top_muts) {
#     if (!feat %in% rownames(mut_binary)) next
#     feat_data <- mut_binary[feat, ]
#     sub_d_feat <- sub_d %>% mutate(Val_Score = as.numeric(feat_data[sample_name])) %>% drop_na(Val_Score)
#     if (sum(sub_d_feat$Val_Score, na.rm=TRUE) == 0) next
#     
#     pval <- tryCatch(suppressWarnings(chisq.test(table(factor(sub_d_feat$Cluster), factor(sub_d_feat$Val_Score, levels=c(0,1)))))$p.value, error=function(e) NA)
#     if (is.na(pval) || pval >= 0.05) next
#     
#     # 统计各个 cluster 的频率
#     clu_stats <- sub_d_feat %>% group_by(Cluster) %>% summarise(v = mean(Val_Score, na.rm=TRUE), .groups="drop") %>% arrange(desc(v))
#     high_clu <- clu_stats$Cluster[1]
#     
#     # 记录到统计表
#     stats_row <- clu_stats %>% mutate(Feature = feat, Category = "Mutation", Cancer = canc, P_Value = pval, Highest_Cluster = high_clu) %>% rename(Score_or_Freq = v)
#     detailed_stats_list[[length(detailed_stats_list) + 1]] <- stats_row
#     
#     p_title <- sprintf("%s\nP=%.2e | C%s Hi", feat, pval, high_clu)
#     p <- ggbarplot(clu_stats %>% mutate(v = v*100), x="Cluster", y="v", fill="Cluster") + 
#       scale_fill_manual(values=cls_colors) + theme_bw() + 
#       labs(title=p_title, x="", y="Freq (%)") + 
#       theme(legend.position="none", plot.title=element_text(size=9, face="bold", hjust=0.5, color="#E41A1C"))
#     plots_mut[[feat]] <- p
#   }
#   
#   # --- 2. 通路 (Pathways) ---
#   for (feat in top_paths) {
#     if (!feat %in% colnames(canc_path_full)) next
#     feat_data <- canc_path_full[, feat]
#     sub_d_feat <- sub_d %>% mutate(Val_Score = as.numeric(feat_data[sample_name])) %>% drop_na(Val_Score)
#     
#     pval <- tryCatch(kruskal.test(Val_Score ~ Cluster, data = sub_d_feat)$p.value, error=function(e) NA)
#     if (is.na(pval) || pval >= 0.05) next
#     
#     clu_stats <- sub_d_feat %>% group_by(Cluster) %>% summarise(v = mean(Val_Score, na.rm=TRUE), .groups="drop") %>% arrange(desc(v))
#     high_clu <- clu_stats$Cluster[1]
#     
#     stats_row <- clu_stats %>% mutate(Feature = feat, Category = "Pathway", Cancer = canc, P_Value = pval, Highest_Cluster = high_clu) %>% rename(Score_or_Freq = v)
#     detailed_stats_list[[length(detailed_stats_list) + 1]] <- stats_row
#     
#     p_title <- sprintf("%s\nP=%.2e | C%s Hi", str_trunc(feat, 20), pval, high_clu)
#     p <- ggviolin(sub_d_feat, x="Cluster", y="Val_Score", fill="Cluster", add="boxplot", add.params=list(fill="white", width=0.1), trim=FALSE) + 
#       scale_fill_manual(values=cls_colors) + theme_bw() + 
#       labs(title=p_title, x="", y="Z-Score") + 
#       theme(legend.position="none", plot.title=element_text(size=9, face="bold", hjust=0.5, color="#377EB8"))
#     plots_path[[feat]] <- p
#   }
#   
#   # --- 3. 免疫细胞 (Immune Cells) ---
#   for (feat in clean_name_fn(top_timers)) {
#     orig_feat <- colnames(canc_timer_full)[clean_name_fn(colnames(canc_timer_full)) == feat][1]
#     if (is.na(orig_feat) || !orig_feat %in% colnames(canc_timer_full)) next
#     feat_data <- canc_timer_full[, orig_feat]
#     sub_d_feat <- sub_d %>% mutate(Val_Score = as.numeric(feat_data[sample_name])) %>% drop_na(Val_Score)
#     
#     pval <- tryCatch(kruskal.test(Val_Score ~ Cluster, data = sub_d_feat)$p.value, error=function(e) NA)
#     if (is.na(pval) || pval >= 0.05) next
#     
#     clu_stats <- sub_d_feat %>% group_by(Cluster) %>% summarise(v = mean(Val_Score, na.rm=TRUE), .groups="drop") %>% arrange(desc(v))
#     high_clu <- clu_stats$Cluster[1]
#     
#     stats_row <- clu_stats %>% mutate(Feature = feat, Category = "Immune_Cell", Cancer = canc, P_Value = pval, Highest_Cluster = high_clu) %>% rename(Score_or_Freq = v)
#     detailed_stats_list[[length(detailed_stats_list) + 1]] <- stats_row
#     
#     p_title <- sprintf("%s\nP=%.2e | C%s Hi", str_trunc(feat, 20), pval, high_clu)
#     p <- ggviolin(sub_d_feat, x="Cluster", y="Val_Score", fill="Cluster", add="boxplot", add.params=list(fill="white", width=0.1), trim=FALSE) + 
#       scale_fill_manual(values=cls_colors) + theme_bw() + 
#       labs(title=p_title, x="", y="Z-Score") + 
#       theme(legend.position="none", plot.title=element_text(size=9, face="bold", hjust=0.5, color="#4DAF4A"))
#     plots_imm[[feat]] <- p
#   }
#   
#   # === 整合并输出单癌症大 PDF ===
#   if (length(plots_mut) > 0 || length(plots_path) > 0 || length(plots_imm) > 0) {
#     pdf_path <- file.path(val_dir_cancer, paste0(canc, "_Significant_Features.pdf"))
#     
#     # 计算排版高度（每列最多 6 个，设定 ncol=6，计算总行数）
#     calc_rows <- function(plist) { if(length(plist)==0) 0 else ceiling(length(plist)/6) }
#     total_rows <- calc_rows(plots_mut) + calc_rows(plots_path) + calc_rows(plots_imm)
#     
#     pdf(pdf_path, width = 20, height = max(6, total_rows * 3.5))
#     
#     arrange_group <- function(plist, title) {
#       if(length(plist) == 0) return(NULL)
#       ggarrange(plotlist = plist, ncol = min(6, length(plist)), nrow = calc_rows(plist)) %>% 
#         annotate_figure(top = text_grob(title, face="bold", size=16, color="black", just="left", x=0.02))
#     }
#     
#     final_arrangements <- list()
#     if(length(plots_mut) > 0) final_arrangements[[1]] <- arrange_group(plots_mut, "Top Significant Mutations")
#     if(length(plots_path) > 0) final_arrangements[[length(final_arrangements)+1]] <- arrange_group(plots_path, "Top Significant Pathways")
#     if(length(plots_imm) > 0) final_arrangements[[length(final_arrangements)+1]] <- arrange_group(plots_imm, "Top Significant Immune Cells")
#     
#     # 纵向拼接三大块
#     print(ggarrange(plotlist = final_arrangements, ncol = 1, heights = c(calc_rows(plots_mut), calc_rows(plots_path), calc_rows(plots_imm))))
#     dev.off()
#   }
# }
# 
# # ==================== 导出全域详细特征分布表 ====================
# if (length(detailed_stats_list) > 0) {
#   detailed_stats_df <- bind_rows(detailed_stats_list) %>%
#     # 数据透视，让各个 Cluster 的均值/频率横向展开，方便阅读
#     pivot_wider(names_from = Cluster, values_from = Score_or_Freq, names_prefix = "Cluster_") %>%
#     select(Cancer, Category, Feature, P_Value, Highest_Cluster, starts_with("Cluster_")) %>%
#     arrange(Cancer, Category, P_Value)
#   
#   stats_csv_path <- file.path(output_dir, "Detailed_Significant_Features_Stats.csv")
#   write_csv(detailed_stats_df, stats_csv_path)
#   message(sprintf("  -> 显著特征详细统计表已成功落盘: %s", stats_csv_path))
# }
# 
# message(paste("\n🎉 全栈流水线大满贯完美收官!\n 1. 景观大图与FDR总表已归档\n 2. 按癌症分类的特征阵列 PDF (每列最多6图) 已落盘至:", val_dir_cancer))
# 
# 
# #########   画一个扇面图
# 
# type1_feature <-c("BRAF", "RNF43", "KMT2D", "MDN1", "CEP350","IL12-mediated_signaling_events" ,"Interleukin-2_signaling",'IL23-mediated_signaling_events',"Interleukin-2_signaling",
#                   "TCR_signaling_in_na_ve_CD8__T_cells","TCR_signaling_in_na_ve_CD4__T_cells","CD28_co-stimulation","PD-1_signaling","MHC_class_II_antigen_presentation",
#                   "granzyme_a_mediated_apoptosis_pathway","caspase_cascade_in_apoptosis","The_NLRP3_inflammasome","The_AIM2_inflammasome","The_IPAF_inflammasome",
#                   "T cell CD8+ XCELL","T cell CD8+ QUANTISEQ","T cell CD8+ CIBERSORT-ABS","T cell CD8+ TIMER","Macrophage M1 QUANTISEQ","Macrophage M1 CIBERSORT","Macrophage M1 XCELL","Myeloid dendritic cell activated XCELL",
#                   "NK cell QUANTISEQ","NK cell EPIC","NK cell activated CIBERSORT-ABS","cytotoxicity score MCPCOUNTER","immune score XCELL")
# type2_feature <-c("CDH1", "PIK3CA", "MAP3K1", "HRAS","Cell-extracellular_matrix_interactions","Molecules_associated_with_elastic_fibres","Elastic_fibre_formation","Crosslinking_of_collagen_fibrils","Degradation_of_collagen","Smooth_Muscle_Contraction","Keratan_sulfate_biosynthesis",
#                   "PDGF_receptor_signaling_network","Signaling_by_PDGF","tgf_beta_signaling_pathway","TGF-beta_receptor_signaling","VEGF_ligand-receptor_interactions",
#                   "Cancer associated fibroblast MCPCOUNTER","Cancer associated fibroblast XCELL","Cancer associated fibroblast EPIC","Endothelial cell MCPCOUNTER","Endothelial cell XCELL","Endothelial cell EPIC","stroma score XCELL","microenvironment score XCELL")
# 
# 
# type3_feature <-c("TP53", "BAP1","Mitotic_Prometaphase","Mitotic_Metaphase/Anaphase_Transition","Separation_of_Sister_Chromatids","Resolution_of_Sister_Chromatid_Cohesion","G1/S-Specific_Transcription","Cyclin_A/B1_associated_events_during_G2/M_transition","Kinesins","E2F_mediated_regulation_of_DNA_replication","E2F_transcription_factor_network","FOXM1_transcription_factor_network","DNA_replication_initiation","Unwinding_of_DNA","Aurora_A_signaling","Aurora_B_signaling",
#                   "uncharacterized cell EPIC","uncharacterized cell QUANTISEQ","Macrophage M0 CIBERSORT","Macrophage M0 CIBERSORT-ABS")
# type4_feature<-c("PTEN", "CTNNB1", "KRAS", "STK11",
#                  "Regulation_of_Cholesterol_Biosynthesis_by_SREBP_(SREBF)","srebp_control_of_lipid_synthesis","Beta_oxidation_of_hexanoyl-CoA_to_butanoyl-CoA","Beta_oxidation_of_butanoyl-CoA_to_acetyl-CoA","Beta-oxidation_of_very_long_chain_fatty_acids","Citric_acid_cycle_(TCA_cycle)", "Fructose_catabolism","Amino_acid_and_oligopeptide_SLC_transporters","wnt_signaling_pathway","Canonical_Wnt_signaling_pathway","Signaling_by_NOTCH3","Notch_signaling_pathway",
#                  "Macrophage M2 CIBERSORT","Macrophage M2 XCELL","Macrophage M2 QUANTISEQ", "Macrophage M2 CIBERSORT-ABS","T cell regulatory (Tregs) CIBERSORT","T cell regulatory (Tregs) CIBERSORT-ABS","T cell regulatory (Tregs) QUANTISEQ","T cell regulatory (Tregs) XCELL")
# 
# 
# cancer_cluster_type <-read.csv("/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section4/section4_cancer_cluster_type.csv")
# 
# 
# 
# library(tidyverse)
# library(circlize)
# library(ComplexHeatmap)
# 
# message("\n[1/5] 正在读取并清洗不规则的亚群分型映射表格...")
# 
# csv_path <- '/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section4/section4_cancer_cluster_type.csv'
# 
# # 强制只取前 3 列，斩断 Excel 幽灵列
# raw_csv <- read.csv(csv_path, stringsAsFactors = FALSE, header = TRUE)[, 1:3]
# colnames(raw_csv) <- c("Cancer", "Cluster", "Archetype")
# 
# # 深度清洗 Cancer 列 
# raw_csv$Cancer <- gsub("\\s*\\(.*?\\)", "", raw_csv$Cancer) 
# raw_csv$Cancer <- trimws(raw_csv$Cancer)
# 
# for(i in 1:nrow(raw_csv)){
#   if(raw_csv$Cancer[i] == "" && i > 1){
#     raw_csv$Cancer[i] <- raw_csv$Cancer[i-1] 
#   }
# }
# 
# type_df <- raw_csv %>%
#   mutate(Archetype = trimws(Archetype)) %>%
#   separate_rows(Cluster, sep = ",") %>%
#   mutate(Cluster = trimws(Cluster)) %>%
#   filter(Cluster != "")
# 
# message("[2/5] 构建用户自定义的手动特征池...")
# 
# all_features <- c(type1_feature, type2_feature, type3_feature, type4_feature)
# 
# feat_lengths <- c(length(type1_feature), length(type2_feature), length(type3_feature), length(type4_feature))
# feat_breaks <- cumsum(feat_lengths)
# 
# message("[3/5] 穿透三大高维矩阵，靶向聚合 Cancer_Cluster 的特征均值...")
# 
# plot_df <- type_df %>% 
#   arrange(Archetype, Cancer, Cluster) %>%
#   mutate(cc_id = paste(Cancer, Cluster, sep = "_"))
# 
# feat_mat <- matrix(NA, nrow = length(all_features), ncol = nrow(plot_df))
# rownames(feat_mat) <- all_features
# colnames(feat_mat) <- plot_df$cc_id
# 
# for(i in 1:nrow(plot_df)) {
#   canc <- plot_df$Cancer[i]
#   clu <- plot_df$Cluster[i]
#   samps <- combined_sorted %>% filter(Cancer == canc, Cluster == clu) %>% pull(sample_name)
#   if(length(samps) == 0) next
#   
#   for(feat in all_features) {
#     if(feat %in% rownames(mut_binary)) {
#       feat_mat[feat, i] <- mean(mut_binary[feat, samps], na.rm=TRUE) 
#     } else if(feat %in% colnames(canc_path_full)) {
#       feat_mat[feat, i] <- mean(canc_path_full[samps, feat], na.rm=TRUE) 
#     } else if(feat %in% colnames(canc_timer_full)) {
#       feat_mat[feat, i] <- mean(canc_timer_full[samps, feat], na.rm=TRUE) 
#     }
#   }
# }
# 
# plot_df <- plot_df %>% group_by(Archetype) %>% mutate(x_local = row_number()) %>% ungroup()
# 
# message("[4/5] 启动 Circlize 环形渲染引擎 (15x15经典比例)...")
# 
# output_pdf_path <- file.path(output_dir, 'Figure4_Archetype_Circos.pdf')
# # 恢复 15x15 的经典物理画布
# pdf(output_pdf_path, width = 15, height = 15)
# 
# circos.clear()
# sectors <- unique(plot_df$Archetype)
# gap.degree <- rep(2, length(sectors))
# # 缺口恢复为 45 度
# gap.degree[length(sectors)] <- 45 
# 
# circos.par("start.degree" = 90, 
#            "gap.degree" = gap.degree, 
#            "track.margin" = c(0.01, 0.01),
#            "cell.padding" = c(0, 0, 0, 0))
# 
# xlim_df <- plot_df %>% count(Archetype) %>% mutate(start = 0, end = n)
# circos.initialize(factors = xlim_df$Archetype, xlim = as.matrix(xlim_df[, c("start", "end")]))
# 
# # --- 轨道 1：Archetype ---
# circos.track(ylim = c(0, 1), track.height = 0.06, bg.border = NA,
#              panel.fun = function(x, y) {
#                sector.index = CELL_META$sector.index
#                xlim = CELL_META$xlim 
#                bg_col <- switch(sector.index, "Type 1 (Immune-Hot)" = "#fee0d2", "Type 2 (Stromal-Rich)" = "#e5f5e0", "Type 3 (Proliferative)" = "#deebf7", "Type 4 (Metabolic)" = "#fde0dd", "#f0f0f0")
#                circos.rect(xlim[1], 0, xlim[2], 1, col = bg_col, border = "black", lwd = 1.2)
#                circos.text(mean(xlim), 0.5, sector.index, facing = "bending.inside", cex = 1.2, font = 2)
#              }
# )
# 
# # --- 轨道 2：癌症标签 ---
# circos.track(ylim = c(0, 1), track.height = 0.04, bg.border = "black",
#              panel.fun = function(x, y) {
#                sector.index = CELL_META$sector.index
#                sub_df <- plot_df[plot_df$Archetype == sector.index, ]
#                for(i in 1:nrow(sub_df)) {
#                  col <- if(exists("cancer_colors") && sub_df$Cancer[i] %in% names(cancer_colors)) cancer_colors[sub_df$Cancer[i]] else "#999999"
#                  circos.rect(i - 1, 0, i, 1, col = col, border = NA)
#                }
#              }
# )
# 
# # --- 轨道 3：Cluster 标签 ---
# circos.track(ylim = c(0, 1), track.height = 0.04, bg.border = "black",
#              panel.fun = function(x, y) {
#                sector.index = CELL_META$sector.index
#                sub_df <- plot_df[plot_df$Archetype == sector.index, ]
#                for(i in 1:nrow(sub_df)) {
#                  circos.rect(i - 1, 0, i, 1, col = "#f7f7f7", border = NA)
#                  circos.text(i - 0.5, 0.5, sub_df$Cluster[i], facing = "inside", cex = 0.65, font=2)
#                }
#              }
# )
# 
# # --- 轨道 4：特征矩阵 ---
# col_mut <- colorRamp2(c(0, 0.4, 0.8), c("#f7fcf5", "#41ab5d", "#00441b")) 
# col_score <- colorRamp2(c(-1.5, 0, 1.5), c("navy", "white", "firebrick3")) 
# num_feats <- length(all_features)
# 
# # 恢复 0.6 的轨道高度比例
# circos.track(ylim = c(0, num_feats), track.height = 0.6, bg.border = "black",
#              panel.fun = function(x, y) {
#                sector.index = CELL_META$sector.index
#                xlim = CELL_META$xlim
#                sub_df <- plot_df[plot_df$Archetype == sector.index, ]
#                
#                for(i in 1:nrow(sub_df)) {
#                  for(j in 1:num_feats) {
#                    feat <- all_features[j]
#                    val <- feat_mat[feat, sub_df$cc_id[i]]
#                    
#                    if(is.na(val)) { cell_col <- "#f0f0f0" } 
#                    else if(feat %in% rownames(mut_binary)) { cell_col <- col_mut(val) } 
#                    else { cell_col <- col_score(val) }
#                    
#                    y_top <- num_feats - j + 1
#                    y_bot <- num_feats - j
#                    circos.rect(i - 1, y_bot, i, y_top, col = cell_col, border = NA)
#                  }
#                }
#                
#                for(brk in feat_breaks[-length(feat_breaks)]) {
#                  circos.lines(x = c(xlim[1], xlim[2]), y = c(num_feats - brk, num_feats - brk), col = "white", lwd = 1.5)
#                }
#                
#                if(sector.index == sectors[length(sectors)]) {
#                  for(j in 1:num_feats) {
#                    feat_clean <- gsub("_", " ", all_features[j]) 
#                    
#                    # ★ 核心改动：字号降为 0.35，并且使用 font = 1 (常规体)，杜绝加粗导致的笔画挤压
#                    circos.text(xlim[2], num_feats - j + 0.5, paste0("  ", feat_clean), 
#                                facing = "inside", adj = c(0, 0.5), cex = 0.35, font = 1, col="black")
#                  }
#                }
#              }
# )
# 
# message("[5/5] 打包图例并渲染输出...")
# 
# lgd_mut <- Legend(title = "Mutation Freq", col_fun = col_mut, direction = "horizontal", title_position = "topcenter", legend_width = unit(4, "cm"))
# lgd_score <- Legend(title = "Pathway/Immune (Z-Score)", col_fun = col_score, direction = "horizontal", title_position = "topcenter", legend_width = unit(4, "cm"))
# 
# if(exists("cancer_colors")) {
#   lgd_cancer <- Legend(title = "Cancer Type", labels = intersect(names(cancer_colors), unique(plot_df$Cancer)), legend_gp = gpar(fill = cancer_colors[intersect(names(cancer_colors), unique(plot_df$Cancer))]), nrow = 3)
#   pd <- packLegend(lgd_mut, lgd_score, lgd_cancer, direction = "vertical")
# } else {
#   pd <- packLegend(lgd_mut, lgd_score, direction = "vertical")
# }
# 
# draw(pd, x = unit(1, "cm"), y = unit(1, "cm"), just = c("left", "bottom"))
# 
# circos.clear()
# dev.off()
# 
# message(paste("\n🎉 经典比例版微缩文字排版完成！高清晰度全景圈图已保存至:\n", output_pdf_path))
# 
# ##########   桑葚图
# 
# # ==============================================================================
# # JEPA Validation Sankey Plots (LGG, BRCA, THCA, UCEC)
# # 核心特性：三级信息流自动追踪 (JEPA Cluster -> Key Mutations -> Classic Subtype)
# # ==============================================================================
# 
#   library(tidyverse)
#   library(ggplot2)
#   library(ggalluvial) # ★ 核心冲积图/桑基图渲染引擎
#   library(RColorBrewer)
# 
# 
# message("\n[1/3] 初始化靶向验证字典与输出目录...")
# 
# # 建立专属的保存文件夹
# sankey_dir <- file.path('/home/Disk30T/chengpeng/Multimodal_JEPA/result/Section_plot/section4', 'Sankey_Validation')
# dir.create(sankey_dir, recursive = TRUE, showWarnings = FALSE)
# 
# # ★ 严格按照你的经典故事总结，配置四大旗舰癌种的“专属靶向突变池”
# target_cancers <- c("LGG", "BRCA", "THCA", "UCEC")
# 
# cancer_story_muts <- list(
#   "LGG"  = c("CIC", "FUBP1", "TP53", "ATRX", "IDH1", "EGFR", "PTEN", "NF1"),
#   "BRCA" = c("CDH1", "PIK3CA", "TP53"),
#   "THCA" = c("BRAF", "HRAS", "NRAS"),
#   "UCEC" = c("POLE", "TP53", "PTEN", "CTNNB1")
# )
# 
# # ==================== Step 2: 核心生信降维法则 ====================
# # 💡 逻辑：真实患者常有几十个并发突变。为了保证图像主干的犀利，
# # 我们依据 TCGA 金标准引入“临床优先级裁决”，提取对分型最具决定性的主驱动基因。
# assign_dominant_mutation <- function(canc, samp) {
#   avail_muts <- cancer_story_muts[[canc]]
#   # 提取该患者在靶向池中命中的突变
#   hits <- avail_muts[sapply(avail_muts, function(m) {
#     if(m %in% rownames(mut_binary)) mut_binary[m, samp] else FALSE
#   })]
#   
#   if (length(hits) == 0) return("WT/Other")
#   
#   if (canc == "LGG") {
#     if (any(c("CIC", "FUBP1") %in% hits)) return("CIC / FUBP1")
#     if (any(c("EGFR", "PTEN", "NF1") %in% hits)) return("EGFR / PTEN / NF1")
#     if (any(c("IDH1", "TP53", "ATRX") %in% hits)) return("IDH1 / TP53 / ATRX")
#   } else if (canc == "BRCA") {
#     if ("CDH1" %in% hits) return("CDH1")     # 小叶癌标志绝对优先
#     if ("TP53" %in% hits) return("TP53")     # 三阴/Basal 驱动
#     if ("PIK3CA" %in% hits) return("PIK3CA") # Luminal 驱动
#   } else if (canc == "THCA") {
#     if ("BRAF" %in% hits) return("BRAF")     # PTC 炎症驱动
#     if (any(c("HRAS", "NRAS") %in% hits)) return("RAS Family") # 归拢 FTC 的 RAS 驱动
#   } else if (canc == "UCEC") {
#     if ("POLE" %in% hits) return("POLE")     # 超突变优先级最高
#     if ("TP53" %in% hits) return("TP53")     # 浆液性高危次之
#     if ("PTEN" %in% hits) return("PTEN")     # 内膜样驱动
#     if ("CTNNB1" %in% hits) return("CTNNB1") # Wnt排斥亚型驱动
#   }
#   
#   return("WT/Other")
# }
# 
# # ==================== Step 3: 提取多级流量并自动渲染 ====================
# message("[2/3] 启动智能桑基图流向映射引擎...")
# 
# # 定义极具学术质感的高对比色卡
# cluster_colors <- c(brewer.pal(8, "Set2"), brewer.pal(8, "Dark2"))
# 
# for(canc in target_cancers) {
#   
#   # 1. 提取底层患者表，丢弃没有经典 Subtype 的模糊样本
#   sub_df <- combined_sorted %>% 
#     filter(Cancer == canc) %>%
#     filter(!is.na(Type) & Type != "Unknown" & Type != "NA" & Type != "")
#   
#   if(nrow(sub_df) == 0) {
#     message(paste("   ⚠️ [跳过]", canc, "- 缺乏可用的临床经典亚型数据。"))
#     next
#   }
#   
#   message(paste("   -> 正在提取与渲染", canc, "的高分辨三级对齐图谱..."))
#   
#   # 2. 挂载经过降维裁决的靶向突变状态
#   sub_df$Driver_Mutation <- sapply(sub_df$sample_name, function(s) assign_dominant_mutation(canc, s))
#   
#   # 3. 统计三级流量 (Cluster -> Mutation -> Subtype)
#   plot_df <- sub_df %>%
#     mutate(
#       Cluster = paste0("C", Cluster), # 加上 "C" 前缀增加美观度
#       Type = str_trunc(Type, 25)      # 截断极度冗长的亚型名防溢出
#     ) %>%
#     group_by(Cluster, Driver_Mutation, Type) %>%
#     summarise(Freq = n(), .groups = "drop") %>%
#     # ★ 防噪截断：过滤掉极其罕见的散发分支连线（< 3人的偶发情况），让主干水流清晰粗壮
#     filter(Freq >= 3)
#   
#   # 动态分配簇颜色
#   unique_clusters <- sort(unique(plot_df$Cluster))
#   canc_cols <- setNames(cluster_colors[1:length(unique_clusters)], unique_clusters)
#   
#   # 4. ggplot2 + ggalluvial 核心渲染引擎
#   p <- ggplot(plot_df,
#               aes(y = Freq, axis1 = Cluster, axis2 = Driver_Mutation, axis3 = Type)) +
#     # 绘制半透明数据水流 (着色锚定为我们的 JEPA Cluster！)
#     geom_alluvium(aes(fill = Cluster), alpha = 0.7, width = 1/5, curve_type = "cubic") +
#     # 绘制承接流量的物理节点柱
#     geom_stratum(width = 1/5, fill = "#f4f4f4", color = "black", linewidth = 0.5) +
#     # 节点智能贴字打标
#     geom_text(stat = "stratum", aes(label = after_stat(stratum)), size = 3.5, fontface = "bold") +
#     # X 轴坐标设置与色彩绑定
#     scale_x_discrete(limits = c("JEPA Cluster", "Driver Mutation", "Clinical Subtype"), expand = c(0.12, 0.12)) +
#     scale_fill_manual(values = canc_cols) +
#     theme_void() +
#     labs(
#       title = paste0(canc, " High-Resolution Alignment"),
#       subtitle = "Tracing the flow from JEPA Clusters through Core Mutations to Classical Subtypes"
#     ) +
#     theme(
#       plot.title = element_text(size = 18, face = "bold", hjust = 0.5, margin = margin(t=15, b=5)),
#       plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40", margin = margin(b = 20)),
#       axis.text.x = element_text(size = 14, face = "bold", color = "black", margin = margin(b=15)),
#       legend.position = "none" # 水流已经自证来源颜色，不需要额外图例
#     )
#   
#   # 5. 高清 PDF 输出
#   pdf_path <- file.path(sankey_dir, paste0("Sankey_Validation_", canc, ".pdf"))
#   pdf(pdf_path, width = 10, height = 7.5)
#   print(p)
#   dev.off()
# }
# 
# message(paste("\n[3/3] 🎉 四大旗舰级金标准验证桑基图渲染完毕！已保存至:\n", sankey_dir))
