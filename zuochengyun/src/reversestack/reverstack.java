package reversestack;

import java.util.Stack;

public class reverstack {
    public  int getAndRemoveLastElement(Stack<Integer> stack){
        int result = stack.pop();
        if (stack.isEmpty()){
            return result;
        }else {
            int last = getAndRemoveLastElement(stack);
            stack.push(result);
            //System.out.println(last);
            return last;

        }
    }

    public  void reverse(Stack<Integer> stack){
        if (stack.isEmpty()){
            return;
        }
        int i = getAndRemoveLastElement(stack);
        System.out.println(i);
        reverse(stack);

        stack.push(i);
    }
}
//递归过程的执行总是一个过程体未执行完，就带着本次执行的结果又进入
// 另一轮过程体的执行，……，如此反复，不断深入，直到某次过程的
// 执行时终止递归调用的条件成立，则不再深入，而执行本次的过程体余下的部分，
// 然后又返回到上一次调用的过程体中，执行余下的部分，……，如此反复，
// 直到回到起始位置上，才最终结束整个递归过程的执行，得到相应的执行结果。
