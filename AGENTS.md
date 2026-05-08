# AGENTS.md

## Project workflow

每次开始一个较大的开发/修改任务前，我会进行plan，请你将最终决定的plan在 `dev/plans/` 目录下创建计划文件：

- 文件名格式：`序号_xxx_plan.md`
- 序号递增，例如：`001_sparse_reader_plan.md`

每次实现完某个plan，后续的针对这个plan的进一步每一次优化或修改后，在 `dev/impls/序号_xxx_impl.md` 中**追加**开发记录。

开发记录不需要很长，可以包含如下内容（建议可以根据情况添加其他内容）：

```md
## YYYY-MM-DD - 简短标题
- 开发目的
- 修改内容
- 影响文件
- 后续注意
```

## 其他

conda环境：wja-cospaq