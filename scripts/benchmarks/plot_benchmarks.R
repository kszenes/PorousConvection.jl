library(ggplot2)
shmem_old_16_8_8 = c(5.851, 2.902, 5.493, 4.540, 3.572) # (16, 8, 8)
shmem = c(5.216, 2.783, 4.794, 4.955, 3.331)        # (32, 4, 4)
original = c(5.551, 3.120, 6.788, 5.524, 3.594)
speedup = (original - shmem) / original * 100

sp2 = sprintf("%0.2f%%", round(rep(speedup, 2), digits=2))

nz = ny = 255
nx = 2 * (nz + 1) - 1
size = nx * ny * nz

bytes = size * 8 * 1e-9

transfer_weights = c(8, 5, 7, 6, 6) * bytes

A_eff = transfer_weights / shmem * 1000

A_eff


# Size: (511, 255, 255)
df <- data.frame(
  kernel=rep(c(
    "flux_P", "P", "flux_T",
    "dTdt", "T"),2),
  Type=c(rep("Shared Memory", 5), rep("Original", 5)),
  timing=c(shmem, original))
  
ggplot(data=df, aes(x=kernel, y=timing, fill=Type), xlab="Kernel") +
  geom_bar(stat="identity", position=position_dodge(), alpha=1) +
  labs(x="Kernel", y="Runtime [ms]", title="Original vs Shared Memory Kernel Runtimes") +
  geom_text(aes(x=kernel, y=1.5, label=ifelse(Type=="Shared Memory", sp2, "")), fontface="bold", colour=ifelse(sp2>0,"green", "red"))


# Block size

block_sizes = c("(32, 4, 4)", "(16, 8, 8)", "(16, 4, 4)", "(8, 8, 8)", "(8, 4, 4)")
time_blocks = c(21.174559000000006, 24.911526000000003, 22.567120999999996, 30.686011000000003, 28.200949000000003)

df_block = data.frame(block_sizes, time_blocks, stringsAsFactors = FALSE)
df_block$block_sizes <- factor(df_block$block_sizes, levels=df_block$block_sizes)

ggplot(data=df_block, aes(x=block_sizes, y=time_blocks)) +
  geom_bar(stat="identity") +
  labs(x="Block Size", y="Runtime [ms]", title="Runtime of Various Block Sizes")

# Kernel T_eff
kernel_names = c("flux_P", "P", "flux_T", "dTdt", "T")
A_eff

df_eff = data.frame(kernel_names, A_eff, stringsAsFactors = FALSE)
#df_eff$kernel_names <- factor(df_eff$kernel_names, levels=df_eff$kernel_names)

ggplot(data=df_eff, aes(x=kernel_names, y=A_eff)) +
  geom_bar(stat="identity") +
  labs(x="Kernel", y="Throughput [GB/s]", title="Effective Throughput of Kernels") +
  geom_text(aes(x=kernel_names, y=A_eff, label=sprintf("%.2f", A_eff)), fontface="bold", vjust=-0.5) +
  ylim(0, 510)

# Weak scaling

# nz = 255
nodes = c(1, 4, 16, 25, 64)
timings_shmem = c(0.02201074956681164, 0.024804903011696012, 0.027018942121668582, 0.02679001999500456, 0.029871598895094463)
norm_shmem <- timings_shmem / timings_shmem[1]

timings_hide = c(0.025814456949028716, 0.026153945572236006, 0.026474373438875386, 0.02646161150735736, 0.026676267882675606)
norm_hide <- timings_hide / timings_hide[1]

df_weak = data.frame(nodes, timings_shmem, norm_timings, timings_hide, norm_hide)

ggplot(data=df_weak, aes(x=nodes)) +
  geom_line(aes(y=timings_shmem, colour="Shared Memory")) +
  geom_line(aes(y=timings_hide, colour="Hide Communication")) +
  geom_point(aes(y=timings_shmem)) +
  geom_point(aes(y=timings_hide)) +
  labs(x="Nodes", y="Runtime/iteration [s]", title="Weak Scaling", colour="Implementation") +
  theme(legend.position="bottom")
