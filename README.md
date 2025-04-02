# TableBench
<div align="left" style="line-height: 1;">
  <a href="" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53%3F?color=green" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="" style="margin: 2px;">
    <img alt="Data License" src="https://img.shields.io/badge/Data_License-CC--BY--SA--4.0-f5de53%3F?color=blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
Official repository for paper "TableBench: A Comprehensive and Complex Benchmark for Table Question Answering"

<p align="left">
    <a href="https://tablebench.github.io//">🏠 Home Page </a> •
    <a href="https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench">📊 Benchmark Data </a> •
    <a href="https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableInstruct">📚 Instruct Data </a> •
    <a href="https://tablebench.github.io/leaderboard.html">🏆 Leaderboard </a> 
</p>


## Table of contents
- [TableBench](#tablebench)
  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Task Examples](#task-examples)
  - [Data](#data)
  - [License](#license)
  - [Citation](#citation)

## Introduction
**TableBench** is a comprehensive and complex benchmark covering **18** fields within four major categories of table question answering (TableQA) capabilitiesm with **886** test samples, which substantially pushes the limits of LLMs in TableQA scenarios.
<p align="center">
<img src="assets/intro_case.png" width="50%" alt="McEval" />
</p>


## Task Examples
<p align="center">
<img src="assets/example.png" width="80%" alt="McEval" />
</p>

<!-- ### Languages
`['AWK','C','CPP','C#','CommonLisp','CoffeeScript','Dart','EmacsLisp','Elixir','Erlang','Fortran','F#','Go','Groovy','Haskell','HTML','Java','JavaScript','JSON','Julia','Kotlin','Lua','Markdown','Pascal','Perl','PHP','PowerShell','Python','R','Racket','Ruby','Rust','Scala','Scheme','Shell','Swift','Tcl','TypeScript','VisualBasic','VimScript']` -->

Refer to our paper for more details. 

## Data
<div align="center">

| **Dataset** |  **Download** |
| :------------: | :------------: |
| TableBench  | [🤗 HuggingFace](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench)   |
| TableInstruct  | [🤗 HuggingFace](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableInstruct)    |

</div>


<!-- ## Usage -->


## License
This code repository is licensed under the [the MIT License](LICENSE-CODE). The use of TableBench data is subject to the [Apache license 2.0](LICENSE-DATA).

## Citation
If you find our work helpful, please use the following citations.
```bibtext
@misc{wu2025tablebenchcomprehensivecomplexbenchmark,
      title={TableBench: A Comprehensive and Complex Benchmark for Table Question Answering}, 
      author={Xianjie Wu and Jian Yang and Linzheng Chai and Ge Zhang and Jiaheng Liu and Xinrun Du and Di Liang and Daixin Shu and Xianfu Cheng and Tianzhen Sun and Guanglin Niu and Tongliang Li and Zhoujun Li},
      year={2025},
      eprint={2408.09174},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.09174}, 
}
```





