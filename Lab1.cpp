#include <iostream>

using namespace std;


int sum(int a, int b, int c) {
  return a + b + c;
}

int divide(int a , int b){
  if(b == 0){
    cout << "Error: Division by zero" << endl;
    return 0;
  }
  return a / b;
}
int main() {
  int a, b, c;
  cout << "Hello World" << endl;
  cout<<"como estas";
  cout << "hola";
  cout << "chau";
  cin >> a, b;
  cout << sum(a, b, c) << endl;
  return 0;
}
