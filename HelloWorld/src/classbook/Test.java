package classbook;

public class Test {
    public static void main(String [] args){
        Book book = new Book("java","miso",60);
        System.out.println("title:"+book.getTitle());
        System.out.println("author:"+book.getAuthor());
        System.out.println("price:"+book.getPrice());
    }
}
