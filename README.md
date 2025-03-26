生成AI（GAN、拡散モデル）
 
生成AIの基本概念

1. 生成対抗ネットワーク（GAN）の基本概念

 GAN（Generative Adversarial Network） は、敵対的生成ネットワークと呼ばれ、「生成者（Generator）」 と 「識別者（Discriminator）」 の2つのニューラルネットワークが互いに競い合う仕組みを持つ。
 
 Generator：ランダムなノイズから本物に近いデータを生成する。
 
 Discriminator：データが本物か偽物かを判定する。この競争によって、生成されるデータの品質が向上する。
 

Stable Diffusionを用いた画像生成
