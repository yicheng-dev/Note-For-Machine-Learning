package main;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;
import java.util.Vector;

public class DecisionTree{

    private static Vector<Sample> samples = new Vector<>();
    private static Vector<AttributeSet> attributeSets = new Vector<>();
    private static Set<String> labelSet = new HashSet<>();
    private static int labelTypeNum;
    private static Node root = new Node();
    private static int nodeId = 0;
    private static int sampleId = 1;

    private static void readFromFile(String filename){
        File inputFile = new File(filename);
        int index = 0;
        try {
            FileInputStream fileInputStream = new FileInputStream(inputFile);
            Scanner scanner = new Scanner(fileInputStream);
            while (scanner.hasNext()) {
                String line = scanner.nextLine();
                String [] data = line.split("\\s+");
                Sample newSample = new Sample();
                newSample.attributes = new Vector<>();
                for (int i = 0; i < data.length - 1; i ++){
                    newSample.attributes.add(new Attribute(i, data[i]));
                }
                newSample.label = data[data.length - 1];
                newSample.id = sampleId++;
                samples.add(newSample);
            }
            fileInputStream.close();
        }catch (IOException e){
            System.out.println("File not found.");
        }
    }

    private static void buildSet(){
        assert samples != null;
        for (int i = 0; i < samples.size(); i++){
            labelSet.add(samples.get(i).label);
        }
        labelTypeNum = labelSet.size();
        for (int j = 0; j < samples.get(0).attributes.size(); j++){
            AttributeSet attributeSet = new AttributeSet();
            attributeSet.id = samples.get(0).attributes.get(j).id;
            attributeSets.add(attributeSet);
        }
        for (int attr = 0; attr < attributeSets.size(); attr++){
            for (int i = 0; i < samples.size(); i++){
                if (attributeSets.get(attr).values != null && attributeSets.get(attr).values.contains(samples.get(i).attributes.get(attr).value)){
                    continue;
                }
                else {
                    attributeSets.get(attr).values.add(samples.get(i).attributes.get(attr).value);
                }
            }
        }
    }

    private static String allSameLabel(Vector<Sample> D){
        for (int i = 0; i < D.size() - 1; i ++){
            //System.out.println("D[" + i + "]'s label: " + D.get(i).label);
            //System.out.println("D[" + (i + 1) + "]'s label: " + D.get(i + 1).label);
            if (!D.get(i).label.equals(D.get(i + 1).label))
                return null;
        }
        return D.get(0).label;
    }

    private static boolean allSameAttribute(Vector<Sample> D, Vector<AttributeSet> A){
        for (int attr = 0; attr < A.size(); attr ++){
            for (int i = 0; i < D.size() - 1; i++){
                if (!D.get(i).attributes.get(A.get(attr).id).value.equals(D.get(i + 1).attributes.get(A.get(attr).id).value))
                    return false;
            }
        }
        return true;
    }

    private static String majorLabel(Vector<Sample> D){
        int [] labelCnt = new int[labelTypeNum];
        for (int i = 0; i < labelTypeNum; i++)
            labelCnt[i] = 0;
        String [] labelMap = new String[labelTypeNum];
        int cnt = 0;
        for (int i = 0; i < D.size(); i++){
            Sample sample = D.get(i);
            boolean newFlag = true;
            for (int j = 0; j < cnt; j++){
                if (labelMap[j].equals(sample.label)){
                    labelCnt[j]++;
                    newFlag = false;
                    break;
                }
            }
            if (newFlag) {
                labelMap[cnt] = sample.label;
                labelCnt[cnt]++;
                cnt++;
            }
        }
        int maxCnt = -1;
        int maxLabel = 0;
        for (int i = 0; i < cnt; i++){
            if (labelCnt[i] > maxCnt){
                maxLabel = i;
                maxCnt = labelCnt[i];
            }
        }
        return labelMap[maxLabel];
    }

    private static double gainDa(Vector<Sample> D, AttributeSet a){
        double gain = 0;
        double ent = 0;
        double totalNum = (double)D.size();
        for (String label : labelSet){
            double num = 0;
            for (Sample sample : D){
                if (sample.label.equals(label)){
                    num += 1.0;
                }
            }
            double p = num / totalNum;
            if (p > 0)
                ent -= p * (Math.log(p) / Math.log(2));
        }
        //System.out.println("Ent(D): " + ent);
        gain += ent;
        for (String attrValue : a.values){
            Vector<Sample> Dv = new Vector<>();
            for (int i = 0; i < D.size(); i++){
                if (D.get(i).attributes.get(a.id).value.equals(attrValue)){
                    Dv.add(D.get(i));
                }
            }
            if (Dv.isEmpty()){
                continue;
            }
            double dvEnt = 0;
            for (String label : labelSet){
                double num = 0;
                for (Sample dvSample : Dv){
                    if (dvSample.label.equals(label)){
                        num += 1.0;
                    }
                }
                double p = num / (double)Dv.size();
                if (p > 0)
                    dvEnt -= p * (Math.log(p) / Math.log(2));
            }
            //System.out.println("dvEnt: " + dvEnt);
            gain -= (double)Dv.size() / totalNum * dvEnt;
        }
        return gain;
    }

    private static double iva(Vector<Sample> D, AttributeSet a){
        double iva = 0;
        for (String attrValue : a.values){
            Vector<Sample> Dv = new Vector<>();
            for (int i = 0; i < D.size(); i++){
                if (D.get(i).attributes.get(a.id).value.equals(attrValue)){
                    Dv.add(D.get(i));
                }
            }
            if (Dv.isEmpty()){
                continue;
            }
            iva -= (double)Dv.size() / (double)D.size() * (Math.log((double)Dv.size() / (double)D.size()) / Math.log(2));
        }
        return iva;
    }

    private static AttributeSet ID3(Vector<Sample> D, Vector<AttributeSet> A){
        double maxGain = -1;
        int bestAttr = -1;
        for (int attr = 0; attr < A.size(); attr ++){
            double gain = gainDa(D, A.get(attr));
            //System.out.println("gain: " + gain);
            if (gain > maxGain){
                maxGain = gain;
                bestAttr = attr;
            }
        }
        return A.get(bestAttr);
    }

    private static AttributeSet C4_5(Vector<Sample> D, Vector<AttributeSet> A){
        double sumGain = 0;
        Vector<AttributeSet> aboveAverageAttrs = new Vector<>();
        double [] gains = new double[A.size()];
        double [] ivs = new double[A.size()];
        for (int attr = 0; attr < A.size(); attr ++){
            gains[attr] = gainDa(D, A.get(attr));
            sumGain += gains[attr];
            ivs[attr] = iva(D, A.get(attr));
        }
        double averageGain = sumGain / A.size();
        for (int attr = 0; attr < A.size(); attr ++){
            if (gains[attr] >= averageGain)
                aboveAverageAttrs.add(A.get(attr));
        }
        //System.out.println(aboveAverageAttrs.size());
        double maxGainRatio = -1;
        int bestAttr = -1;
        for (int attr = 0; attr < aboveAverageAttrs.size(); attr++){
            double gainRatio = gainDa(D, aboveAverageAttrs.get(attr)) / iva(D, aboveAverageAttrs.get(attr));
            if (gainRatio > maxGainRatio){
                maxGainRatio = gainRatio;
                bestAttr = attr;
            }
        }
        return A.get(bestAttr);
    }

    private static void treeGenerate(Vector<Sample> D, Vector<AttributeSet> A, Node node, String algorithm){
        /*System.out.println("treeGenerate begins ---\n\tNum of samples: " + D.size() + "\n\tNum of attributes: " + A.size());
        System.out.print("\tD: ");
        for (int i = 0; i < D.size(); i++){
            System.out.print(D.get(i).id + " ");
        }
        System.out.println();*/
        if (allSameLabel(D) != null){
            //System.out.println("\tall same label!");
            node.label = allSameLabel(D);
            node.type = 2;
            return;
        }
        if (A.isEmpty() || allSameAttribute(D, A)){
            /*if (A.isEmpty())
                System.out.println("\tA is empty.");
            else if (allSameAttribute(D, A))
                System.out.println("\tall same attribute!");*/
            node.label = majorLabel(D);
            node.type = 2;
            return;
        }
        AttributeSet bestAttr;
        switch (algorithm){
            case "ID3":
                bestAttr = ID3(D, A);
                //System.out.println("\tbestAttr: " + bestAttr.id);
                break;
            case "C4.5":
                default:
                bestAttr = C4_5(D, A);
                break;
        }
        //System.out.println("\tNow bestAttr: " + bestAttr.id + " values: " + bestAttr.values);
        for (String attrValue : bestAttr.values){
            Node newNode = new Node();
            newNode.parent = node;
            newNode.type = 1;
            newNode.id = nodeId++;
            node.children.add(newNode);
            Vector<Sample> Dv = new Vector<>();
            for (int i = 0; i < D.size(); i++){
                if (D.get(i).attributes.get(bestAttr.id).value.equals(attrValue)){
                    Dv.add(D.get(i));
                }
            }
            if (Dv.isEmpty()){
                //System.out.println("Dv is empty!");
                newNode.type = 2;
                newNode.label = majorLabel(D);
                return;
            }
            else{
                Vector<AttributeSet> Av = new Vector<>();
                for (int i = 0; i < A.size(); i++){
                    if (A.get(i) != bestAttr)
                        Av.add(A.get(i));
                }
                treeGenerate(Dv, Av, newNode, algorithm);
            }
        }
    }

    private static void printTree(Node node, int space){
        for (int i = 0; i < space; i++)
            System.out.print(" ");
        System.out.print(node.id);
        if (node.type == 2){
            System.out.println("(" + node.label + ")");
        }
        else{
            System.out.println();
        }
        if (node.type != 2){
            for (int i = 0; i < node.children.size(); i++){
                printTree(node.children.get(i), space + 4);
            }
        }
    }

    public static void generate(String filename, String algorithm){
        samples.clear();
        attributeSets.clear();
        labelSet.clear();
        labelTypeNum = 0;
        root = new Node();
        nodeId = 0;
        sampleId = 1;
        readFromFile(filename);
        buildSet();
        root.type = 0;
        root.id = nodeId++;
        treeGenerate(samples, attributeSets, root, algorithm);
        printTree(root, 0);
    }

    public static void main(String[] args){
        generate("data.txt", "ID3");
        generate("data.txt", "C4.5");
    }
}

class Attribute{
    int id;
    String value;
    Attribute(int id, String value){
        this.id = id;
        this.value = value;
    }
}

class AttributeSet{
    int id;
    Vector<String> values = new Vector<>();
}

class Sample{
    int id;
    Vector<Attribute> attributes;
    String label;
}

class Node{
    int id;
    Node parent;
    Vector<Node> children = new Vector<>();
    int type; // 0 - root; 1 - mid; 2 - leaf
    String label;
}
