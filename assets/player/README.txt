玩家立绘预设（标注贴图优先用各目录 stand1_0.png；战斗类用 swing*/stab*/shoot*）

详见 docs/如何抽取玩家精灵图.md

默认男/女新手已内置木剑(1302000)；站立/走路/攻击帧均为持武器 IO 合成立绘。
其他职业预设（男战士/魔法师/弓箭手/飞侠）亦自带对应武器。

复现:
  python scripts/extract_sprites.py --player
  python scripts/extract_sprites.py --player 默认女新手 男战士
