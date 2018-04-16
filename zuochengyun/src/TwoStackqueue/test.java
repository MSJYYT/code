package TwoStackqueue;

public class test {
    public static void main(String [] args){
        TwoStacksQueue twoStacksQueue = new TwoStacksQueue();

        twoStacksQueue.add(1);
        twoStacksQueue.add(2);
        twoStacksQueue.add(3);
        twoStacksQueue.add(4);
        twoStacksQueue.add(5);
        System.out.println(twoStacksQueue.poll());
        System.out.println(twoStacksQueue.poll());
        //System.out.println(twoStacksQueue.peek());
    }
}
