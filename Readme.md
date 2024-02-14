## 💻 安裝

### Git 安裝

> 本倉庫同時上傳到 [Gitee](https://gitee.com/immyw/hexo-theme-butterfly.git)，如果你訪問 Github 緩慢，可從 Gitee 中下載。

在博客根目錄裡安裝穩定版【推薦】

```powershell
git clone -b master https://github.com/jerryc127/hexo-theme-butterfly.git themes/butterfly
```

如果想要安裝比較新的dev分支，可以

```powershell
git clone -b dev https://github.com/jerryc127/hexo-theme-butterfly.git themes/butterfly
```

### npm 安裝

> 此方法只支持Hexo 5.0.0以上版本

在博客根目錄裡

```powershell
npm i hexo-theme-butterfly
```

## ⚙ 應用主題

修改hexo配置文件`_config.yml`，把主題改為`Butterfly`

```
theme: butterfly
```

>如果你沒有pug以及stylus的渲染器，請下載安裝： npm install hexo-renderer-pug hexo-renderer-stylus --save