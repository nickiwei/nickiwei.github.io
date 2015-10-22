---
layout:     post
title:      "欢迎访问我的博客"
subtitle:   " \"Welcome to my blog.\""
date:       2015-10-22 12:00:00
author:     "nickiwei"
header-img: "img/post-bg-2015.jpg"
tags:
    - 杂谈
---

> “Yeah It's on. ”


## 前言


[跳到技术实现 ](#build) 

终于把博客挂出来了， 以前托管在公共博客提供商那里， 可定制程度低， 界面丑， 广告还越来越多。 github pages解决了低可定制性的问题， 配合github上浓厚的开源氛围， 非常适合搭建博客， 同时， 也可以fork诸位前端大神精美的作品， 我等后端狗就只好恬不知耻的套模板了。

以前在博客园的文章看了看， 略显幼稚， 决定就觉得不搬过来了。 如果有不错的， 也会考虑修改完善后重新发布。所以基本上就是重头开始了。

最后， 希望大家在这里玩的开心， 能够真的提供一些有用的干货给大家。

---

## 正文

以下是技术实现。

<p id = "build"></p>

博客使用了git/github+jekyll+markdown实现， 使用了开源的jekyll模板， 感谢<a href="http://huangxuan.me">Hux</a>贡献开源模板， 非常好看。

使用了bower+npm+grunt进行前端代码开发环境的管理， 使用ruby/gem+bundle+jekyll搭建本地测试环境，当然， 使用git/github进行版本管理。

目前博客全部代码及资源托管在<a href="https://github.com/nickiwei/nickiwei.github.io">我的Github</a>上， 欢迎大家访问， fork及使用。

内容创作上， 鉴于jekyll之流的静态网页生成器屏蔽了动态访问数据库， WEB后台负载均衡等这些对于博客过于牛刀的模块（当然， 牺牲是每一次访问都需要重新扫描整个文件系统），再加上markdown cover了几乎全部排版相关的工作， 基本上一次架构完成后， 几乎没有任何后期的运营成本， 可以全心全意的放在内容上了。


