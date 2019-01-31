## BayesClassifier

### 简介

 `BayesClassifier.java`是个人实现的贝叶斯分类器，包含朴素贝叶斯分类器（NB）和AODE半朴素贝叶斯分类器。NB支持离散、连续型属性变量，并且支持拉普拉斯修正。AODE仅支持离散型属性变量，强制拉普拉斯修正。

### 接口说明  
	
传入训练集数据及标签，指明各属性的连续性，开始训练。

```java
public int train(Vector<Vector<String>> samples, Vector<Boolean> continuous, Vector<String> labels, boolean lapCorr, int algorithm);
```

传入测试数据，开始测试，最后返回测试结果。

```java
public String test(Vector<String> testSample);
```

### 使用示例

```java
Vector<Vector<String>> samples = new Vector<>();
Vector<String> labels = new Vector<>();
Vector<Boolean> continuous = new Vector<>();
Vector<String> testSample = new Vector<>();
        
readFromFile("data.txt", samples, labels);
Boolean [] contArray = {false, false, false, false, false, false, true, true};
Collections.addAll(continuous, contArray);
BayesClassifier bc = new BayesClassifier();
if (bc.train(samples, continuous, labels, true) < 0)
	System.out.println("Train failed.");
String [] testSampleArray = {"青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "0.697", "0.460"};
Collections.addAll(testSample, testSampleArray);
String classifyResult = bc.test(testSample);
System.out.println(classifyResult);
```

### 上述示例对应的训练集

```
青绿  蜷缩  浊响  清晰  凹陷  硬滑  0.697 0.460 是  
乌黑  蜷缩  沉闷  清晰  凹陷  硬滑  0.774 0.376 是  
乌黑  蜷缩  浊响  清晰  凹陷  硬滑  0.634 0.264 是  
青绿  蜷缩  沉闷  清晰  凹陷  硬滑  0.608 0.318 是  
浅白  蜷缩  浊响  清晰  凹陷  硬滑  0.556 0.215 是  
青绿  稍蜷  浊响  清晰  稍凹  软粘  0.403 0.237 是  
乌黑  稍蜷  浊响  稍糊  稍凹  软粘  0.481 0.149 是  
乌黑  稍蜷  浊响  清晰  稍凹  硬滑  0.437 0.211 是  
乌黑  稍蜷  沉闷  稍糊  稍凹  硬滑  0.666 0.091 否  
青绿  硬挺  清脆  清晰  平坦  软粘  0.243 0.267 否  
浅白  硬挺  清脆  模糊  平坦  硬滑  0.245 0.057 否  
浅白  蜷缩  浊响  模糊  平坦  软粘  0.343 0.099 否  
青绿  稍蜷  浊响  稍糊  凹陷  硬滑  0.639 0.161 否  
浅白  稍蜷  沉闷  稍糊  凹陷  硬滑  0.657 0.198 否  
乌黑  稍蜷  浊响  清晰  稍凹  软粘  0.360 0.370 否  
浅白  蜷缩  浊响  模糊  平坦  硬滑  0.593 0.042 否  
青绿  蜷缩  沉闷  稍糊  稍凹  硬滑  0.719 0.103 否  
```
