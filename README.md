# Guia de Testes: Performance e Concorrência

Este guia detalha como executar os testes de carga (JMeter + JFR), microbenchmarks (JMH) e testes de estresse de concorrência (JCStress).

## 1. Testes de Carga com JMeter e JFR
Utilize o script `run_jmeter.sh` (baseado no seu arquivo de automação). Ele gera um plano de teste JMX, configura o JFR (Java Flight Recorder) e executa o teste.

### Como rodar:
O comando aceita 4 parâmetros: `Versão`, `GC`, `Threads` e `Loops`.

```bash
# Exemplo: Rodando o teste da versão 'Sync' com G1GC, 20 threads e 5 loops
./run_jmeter.sh Sync G1GC 20 5
```

# Microbenchmarks com JMH
Compile o projeto, execute o `jar` gerado:

```bash
mvn clean package
java -jar target/benchmarks.jar -f 1 -wi 3 -i 5
```

# JCStress
Compile e execute o harness do JCStress:

```bash
mvn clean install
java -jar benchmarks.jar -t KNNConcurrencyStressTest
```
