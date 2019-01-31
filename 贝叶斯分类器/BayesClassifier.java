package main;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class BayesClassifier {
    private Vector<TrainSample> trainSet;
    private int numOfTrainData;
    private int numOfAttributes;
    private Vector<Vector<Attribute>> attributeSet;
    private Vector<Label> labels;
    private int typeNumOfLabels;

    private int algorithm; // 0 represents naive Bayes Classifier; 1 represents AODE.
    private boolean lapCorr;
    private boolean hasTrained;

    private Vector<Double> pc;

    public BayesClassifier(){
        trainSet = new Vector<>();
        attributeSet = new Vector<>();
        labels = new Vector<>();
        lapCorr = true;
        typeNumOfLabels = 0;
        algorithm = 0;
        hasTrained = false;
    }

    public int train(Vector<Vector<String>> samples, Vector<Boolean> continuous, Vector<String> labels){
        if (assertValidation(samples, continuous, labels) < 0) {
            usage();
            return -1;
        }
        buildTrainSet(samples, continuous, labels);
        computePc();
        hasTrained = true;
        return 0;
    }

    public int train(Vector<Vector<String>> samples, Vector<Boolean> continuous, Vector<String> labels, boolean lapCorr){
        this.lapCorr = lapCorr;
        return this.train(samples, continuous, labels);
    }

    public int train(Vector<Vector<String>> samples, Vector<Boolean> continuous, Vector<String> labels, int algorithm){
        if (algorithm != 0 && algorithm != 1){
            System.out.println("The 'algorithm' should be 0 (naive) or be 1 (AODE).");
            return -1;
        }
        this.algorithm = algorithm;
        return this.train(samples, continuous, labels);
    }

    public int train(Vector<Vector<String>> samples, Vector<Boolean> continuous, Vector<String> labels, boolean lapCorr, int algorithm){
        this.lapCorr = lapCorr;
        return this.train(samples, continuous, labels, algorithm);
    }

    public String test(Vector<String> testSample){
        if (!hasTrained){
            System.out.println("Please train the classifier firstly.");
            return null;
        }
        if (assertTestValidation(testSample) < 0){
            return null;
        }
        if (algorithm == 0)
            return naiveBayesJudge(testSample);
        else
            return AODE(testSample);
    }

    private void usage(){
        System.out.println(
                "[usage] ----------\n" +
                "   1. Constructor: BayesClassifier bayesClassifier = new BayesClassifier();\n" +
                "   2. Training: bayesClassifier.train(vec1, vec2, vec3, lapCorr = true, algorithm = 0) ----------\n" +
                "       vec1: Two-dimension String Vector of samples with attributes.\n" +
                "       vec2: Boolean Vector which represents the continuity of each attribute.\n" +
                "       vec3: String Vector of labels corresponding to samples.\n" +
                "       lapCorr: 1 -> have Laplacian Correction; 0 -> have no Laplacian Correction. Default value is 1.\n" +
                "       algorithm: 0 -> naive Bayes Classification(NB); 1 -> semi-naive Bayes Classification(AODE). Default value is 0.\n" +
                "   3. Testing: bayesClassifier.test(vec) ---------\n" +
                "       vec: String vector of a testing sample with attributes.");
    }

    private boolean enoughData(Vector<String> testSample){
        return true;
    }

    private String AODE(Vector<String> testSample){
        double maxEstimate = 0;
        String retStr = "";
        for (int labelIndex = 0; labelIndex < typeNumOfLabels; labelIndex++){
            double averagedEstimate = 0;
            for (int attrIndex = 0; attrIndex < testSample.size(); attrIndex++){
                if (!enoughData(testSample)){
                    continue;
                }
                double currEstimate = 1;
                if (!attributeSet.get(attrIndex).get(0).continuous){
                    int satisCiNum = 0;
                    for (int sampleIndex = 0; sampleIndex < trainSet.size(); sampleIndex++){
                        TrainSample sample = trainSet.get(sampleIndex);
                        if (labels.get(sampleIndex).type == labelIndex && sample.attributes.get(attrIndex).value.equals(testSample.get(attrIndex)))
                            satisCiNum++;
                    }
                    currEstimate *= ((double)satisCiNum + 1) / (numOfTrainData + typeNumOfLabels * attributeSet.get(attrIndex).size());
                    for (int attrIndex2 = 0; attrIndex2 < testSample.size(); attrIndex2++){
                        if (!attributeSet.get(attrIndex2).get(0).continuous){
                            int satisCijNum = 0;
                            for (int sampleIndex = 0; sampleIndex < trainSet.size(); sampleIndex++){
                                TrainSample sample = trainSet.get(sampleIndex);
                                if (labels.get(sampleIndex).type == labelIndex && sample.attributes.get(attrIndex).value.equals(testSample.get(attrIndex))
                                    && sample.attributes.get(attrIndex2).value.equals(testSample.get(attrIndex2)))
                                    satisCijNum++;
                            }
                            currEstimate *= ((double)satisCijNum + 1) / ((double)satisCiNum + attributeSet.get(attrIndex2).size());
                        }
                        else{
                            //TODO: support continuous variables
                        }
                    }
                }
                else{
                    //TODO: support continuous variables
                }
                averagedEstimate += currEstimate;
            }
            if (labelIndex == 0){
                maxEstimate = averagedEstimate;
                for (int i = 0; i < labels.size(); i++){
                    if (labels.get(i).type == labelIndex){
                        retStr = labels.get(i).value;
                        break;
                    }
                }
            }
            else if (averagedEstimate > maxEstimate){
                maxEstimate = averagedEstimate;
                for (int i = 0; i < labels.size(); i++){
                    if (labels.get(i).type == labelIndex){
                        retStr = labels.get(i).value;
                        break;
                    }
                }
            }
        }
        return retStr;
    }

    private String naiveBayesJudge(Vector<String> testSample){
        double maxLog = 0;
        String retStr = "";
        for (int labelIndex = 0; labelIndex < typeNumOfLabels; labelIndex++){
            double currLog = 0.0;
            currLog += Math.log(pc.get(labelIndex));
            for (int attrIndex = 0; attrIndex < testSample.size(); attrIndex++){
                if (!attributeSet.get(attrIndex).get(0).continuous) {
                    int satisNum = 0;
                    for (int sampleIndex = 0; sampleIndex < trainSet.size(); sampleIndex++) {
                        TrainSample sample = trainSet.get(sampleIndex);
                        if (sample.attributes.get(attrIndex).value.equals(testSample.get(attrIndex)) && labels.get(sampleIndex).type == labelIndex)
                            satisNum++;
                    }
                    if (lapCorr)
                        currLog += Math.log(((double)satisNum + 1) / ((double)pc.get(labelIndex) * numOfTrainData + attributeSet.get(attrIndex).size()));
                    else
                        currLog += Math.log(((double) satisNum / (pc.get(labelIndex) * numOfTrainData)));
                }
                else{
                    double sum = 0;
                    Vector<Double> satisVec = new Vector<>();
                    for (int sampleIndex = 0; sampleIndex < trainSet.size(); sampleIndex++) {
                        if (labels.get(sampleIndex).type == labelIndex){
                            double value = Double.parseDouble(trainSet.get(sampleIndex).attributes.get(attrIndex).value);
                            satisVec.add(value);
                            sum += value;
                        }
                    }
                    double mu = sum / satisVec.size();
                    double squareSum = 0;
                    for (double satisValue : satisVec){
                        squareSum += (satisValue - mu) * (satisValue - mu);
                    }
                    double sigma = squareSum / (satisVec.size() - 1);
                    currLog += Math.log((1 / (Math.sqrt(2 * Math.PI) * Math.sqrt(sigma))) * Math.exp(-(Double.parseDouble(testSample.get(attrIndex)) - mu)
                            * (Double.parseDouble(testSample.get(attrIndex)) - mu) / (2 * sigma)));
                }
            }
            if (labelIndex == 0){
                maxLog = currLog;
                for (int i = 0; i < labels.size(); i++){
                    if (labels.get(i).type == labelIndex){
                        retStr = labels.get(i).value;
                        break;
                    }
                }
            }
            else if (currLog > maxLog){
                maxLog = currLog;
                for (int i = 0; i < labels.size(); i++){
                    if (labels.get(i).type == labelIndex){
                        retStr = labels.get(i).value;
                        break;
                    }
                }
            }
        }
        return retStr;
    }

    private void computePc(){
        pc = new Vector<>();
        for (int i = 0; i < typeNumOfLabels; i++){
            int sampleNum = 0;
            for (int j = 0; j < labels.size(); j++){
                if (labels.get(j).type == i)
                    sampleNum++;
            }
            if (lapCorr)
                pc.add(((double)sampleNum + 1) / ((double)numOfTrainData + typeNumOfLabels));
            else
                pc.add((double)sampleNum / (double)numOfTrainData);
        }
    }

    private int assertTestValidation(Vector<String> testSample){
        if (testSample == null || testSample.isEmpty()){
            System.out.println("[failed] 'testsample' shouldn't be empty.");
            return -1;
        }
        if (testSample.size() != numOfAttributes){
            System.out.println("[failed] The size of 'testSample' should be the same of 'continuous'.");
            return -1;
        }
        if (invalidTestCont(testSample) < 0){
            System.out.println("[failed] There is some 'attributes' which aren't numbers but judged as continuous variables.");
            return -1;
        }
        return 0;
    }

    private int assertValidation(Vector<Vector<String>> samples, Vector<Boolean> continuous, Vector<String> labels){
        if (samples == null || samples.isEmpty() || continuous == null || continuous.isEmpty() || labels == null || labels.isEmpty()){
            System.out.println("[failed] 'Samples', 'continuous' or 'labels' should not be empty.");
            return -1;
        }
        if (samples.size() != labels.size()){
            System.out.println("[failed] Size of 'samples' and 'labels' should be all the same.");
            return -1;
        }
        if (samples.get(0).size() != continuous.size()){
            System.out.println("[failed] Size of the first sample should be the same as the size of 'continuous'.");
            return -1;
        }
        if (invalidCont(samples, continuous) < 0){
            System.out.println("[failed] There is some 'attributes' which aren't numbers but judged as continuous variables.");
            return -1;
        }
        if (alignSample(samples) < 0){
            System.out.println("[failed] 'Samples' should be a matrix where each row has the same number of column.");
            return -1;
        }
        return 0;
    }

    private int alignSample(Vector<Vector<String>> samples){
        int colNum = samples.get(0).size();
        for (Vector<String> sample : samples){
            if (sample.size() != colNum)
                return -1;
        }
        return 0;
    }

    private int invalidTestCont(Vector<String> testSample){
        String numberRegex = "[+-]?[0-9]+|[+-]?([0-9]*\\.[0-9]+|[0-9]+\\.[0-9]*)|[+-]?([0-9]*\\.?[0-9]+|[0-9]+\\.?[0-9]*)[Ee][+-]?[0-9]+";
        Pattern numberPat = Pattern.compile(numberRegex);
        for (int i = 0; i < attributeSet.size(); i++){
            if (attributeSet.get(i).get(0).continuous){
                Matcher numberMat = numberPat.matcher(testSample.get(i));
                if (!numberMat.matches())
                    return -1;
            }
        }
        return 0;
    }

    private int invalidCont(Vector<Vector<String>> samples, Vector<Boolean> continuous){
        String numberRegex = "[+-]?[0-9]+|[+-]?([0-9]*\\.[0-9]+|[0-9]+\\.[0-9]*)|[+-]?([0-9]*\\.?[0-9]+|[0-9]+\\.?[0-9]*)[Ee][+-]?[0-9]+";
        Pattern numberPat = Pattern.compile(numberRegex);
        for (int i = 0; i < continuous.size(); i++){
            boolean cont = continuous.get(i);
            for (Vector<String> sample : samples){
                for (int j = 0; j < sample.size(); j++){
                    if (i == j && cont){
                        String attr = sample.get(j);
                        Matcher numberMat = numberPat.matcher(attr);
                        if (!numberMat.matches())
                            return -1;
                    }
                }
            }
        }
        return 0;
    }

    private void buildTrainSet(Vector<Vector<String>> samples, Vector<Boolean> continuous, Vector<String> labels){
        buildLabels(labels);
        numOfTrainData = samples.size();
        for (int i = 0; i < samples.size(); i++){
            Vector<String> sample = samples.get(i);
            TrainSample trainSample = new TrainSample();
            trainSample.label = this.labels.get(i);
            if (i == 0)
                numOfAttributes = sample.size();
            for (int j = 0; j < sample.size(); j++){
                Attribute attribute = new Attribute();
                attribute.value = sample.get(j);
                attribute.id = j;
                attribute.continuous = continuous.get(j);
                trainSample.attributes.add(attribute);
            }
            trainSet.add(trainSample);
        }
        buildAttributeSet();
    }

    private void buildAttributeSet(){
        for (int i = 0; i < numOfAttributes; i++){
            Vector<Attribute> attributes = new Vector<>();
            Set<String> value = new HashSet<>();
            for (TrainSample sample : trainSet){
                if (!value.isEmpty()){
                    if (!value.contains(sample.attributes.get(i).value)){
                        attributes.add(sample.attributes.get(i));
                        value.add(sample.attributes.get(i).value);
                    }
                }
                else {
                    attributes.add(sample.attributes.get(i));
                    value.add(sample.attributes.get(i).value);
                }
            }
            attributeSet.add(attributes);
        }
    }

    private void buildLabels(Vector<String> labels){
        for (String label : labels){
            Label newLabel = new Label(label);
            if (!this.labels.isEmpty()){
                boolean flag = false;
                for (Label oldLabel : this.labels){
                    if (oldLabel.value.equals(newLabel.value)){
                        newLabel.type = oldLabel.type;
                        flag = true;
                        break;
                    }
                }
                if (!flag){
                    newLabel.type = typeNumOfLabels++;
                }
            }
            else{
                newLabel.type = typeNumOfLabels++;
            }
            this.labels.add(newLabel);
        }
    }

    public static void main(String[] args){
        Vector<Vector<String>> samples = new Vector<>();
        Vector<String> labels = new Vector<>();
        Vector<Boolean> continuous = new Vector<>();
        Vector<String> testSample = new Vector<>();

        readFromFile("data.txt", samples, labels);
        Boolean [] contArray = {false, false, false, false, false, false, true, true};
        Collections.addAll(continuous, contArray);
        BayesClassifier nbc = new BayesClassifier();
        if (nbc.train(samples, continuous, labels, true, 1) < 0)
            System.out.println("Train failed.");
        String [] testSampleArray = {"青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "0.697", "0.460"};
        Collections.addAll(testSample, testSampleArray);
        String classifyResult = nbc.test(testSample);
        System.out.println(classifyResult);
    }

    private static void readFromFile(String filename, Vector<Vector<String>> samples, Vector<String> labels){
        File inputFile = new File(filename);
        samples.clear();
        labels.clear();
        try {
            FileInputStream fileInputStream = new FileInputStream(inputFile);
            Scanner scanner = new Scanner(fileInputStream);
            while (scanner.hasNext()) {
                String line = scanner.nextLine();
                String [] data = line.split("\\s+");
                int sampleLabelNum = data.length;
                Vector<String> sample = new Vector<>();
                for (int i = 0; i < sampleLabelNum - 1; i++){
                    sample.add(data[i]);
                }
                samples.add(sample);
                labels.add(data[sampleLabelNum - 1]);
            }
            fileInputStream.close();
        }catch (IOException e){
            System.out.println("File not found.");
        }
    }

}

class TestSample{
    Vector<Attribute> attributes;
    TestSample(){
        attributes = new Vector<>();
    }
}

class TrainSample extends TestSample{
    Label label;
    TrainSample(){
        super();
    }
}

class Attribute{
    int id;
    boolean continuous;
    String value;
}

class Label{
    int type;
    String value;
    Label(String value){
        this.value = value;
    }
}
