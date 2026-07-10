package benchmark;

import org.openjdk.jcstress.annotations.*;
import org.openjdk.jcstress.infra.results.L_Result;
import org.openjdk.jcstress.infra.results.I_Result;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.Semaphore;
import java.util.concurrent.StructuredTaskScope;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

public class KNNConcurrencyStressTest {

    public record DistanceRecord(double distance, String label) {}

    @JCStressTest
    @Description("Simula a ausência de sincronização. Deve apresentar Race Conditions.")
    @Outcome(id = "B:3.0", expect = Expect.ACCEPTABLE, desc = "Thread 2 venceu de forma limpa.")
    @Outcome(id = "A:5.0", expect = Expect.ACCEPTABLE, desc = "Thread 1 venceu de forma limpa.")
    @Outcome(id = "A:3.0", expect = Expect.ACCEPTABLE_INTERESTING, desc = "BUG! Rótulo A com distância B!")
    @Outcome(id = "B:5.0", expect = Expect.ACCEPTABLE_INTERESTING, desc = "BUG! Rótulo B com distância A!")
    @State
    public static class UnsafeState {
        double bestDist = 10.0;
        String bestLabel = "None";

        @Actor
        public void actor1() {
            if (5.0 < bestDist) {
                bestDist = 5.0;
                bestLabel = "A";
            }
        }

        @Actor
        public void actor2() {
            if (3.0 < bestDist) {
                bestDist = 3.0;
                bestLabel = "B";
            }
        }

        @Arbiter
        public void arbiter(L_Result r) {
            r.r1 = bestLabel + ":" + bestDist;
        }
    }

    @JCStressTest
    @Description("Testa a proteção via bloco synchronized (KNNSync).")
    @Outcome(id = {"B:3.0", "A:5.0"}, expect = Expect.ACCEPTABLE, desc = "Sincronização OK.")
    @Outcome(id = {"A:3.0", "B:5.0"}, expect = Expect.FORBIDDEN, desc = "FALHA NA SINCRONIZAÇÃO!")
    @State
    public static class SyncState {
        double bestDist = 10.0;
        String bestLabel = "None";
        private final Object lock = new Object();

        @Actor
        public void actor1() {
            synchronized(lock) {
                if (5.0 < bestDist) { bestDist = 5.0; bestLabel = "A"; }
            }
        }

        @Actor
        public void actor2() {
            synchronized(lock) {
                if (3.0 < bestDist) { bestDist = 3.0; bestLabel = "B"; }
            }
        }

        @Arbiter
        public void arbiter(L_Result r) {
            r.r1 = bestLabel + ":" + bestDist;
        }
    }

    @JCStressTest
    @Description("Testa a proteção via ReentrantLock (KNNLock).")
    @Outcome(id = {"B:3.0", "A:5.0"}, expect = Expect.ACCEPTABLE, desc = "Lock OK.")
    @Outcome(id = {"A:3.0", "B:5.0"}, expect = Expect.FORBIDDEN, desc = "FALHA NO LOCK!")
    @State
    public static class LockState {
        double bestDist = 10.0;
        String bestLabel = "None";
        private final ReentrantLock lock = new ReentrantLock();

        @Actor
        public void actor1() {
            lock.lock();
            try {
                if (5.0 < bestDist) { bestDist = 5.0; bestLabel = "A"; }
            } finally { lock.unlock(); }
        }

        @Actor
        public void actor2() {
            lock.lock();
            try {
                if (3.0 < bestDist) { bestDist = 3.0; bestLabel = "B"; }
            } finally { lock.unlock(); }
        }

        @Arbiter
        public void arbiter(L_Result r) {
            r.r1 = bestLabel + ":" + bestDist;
        }
    }

    @JCStressTest
    @Description("Testa a proteção via Mutex com Semaphore (KNNSemaphore).")
    @Outcome(id = {"B:3.0", "A:5.0"}, expect = Expect.ACCEPTABLE, desc = "Semáforo OK.")
    @Outcome(id = {"A:3.0", "B:5.0"}, expect = Expect.FORBIDDEN, desc = "FALHA NO SEMÁFORO!")
    @State
    public static class SemaphoreState {
        double bestDist = 10.0;
        String bestLabel = "None";
        private final Semaphore mutex = new Semaphore(1);

        @Actor
        public void actor1() {
            try {
                mutex.acquire();
                if (5.0 < bestDist) { bestDist = 5.0; bestLabel = "A"; }
            } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            finally { mutex.release(); }
        }

        @Actor
        public void actor2() {
            try {
                mutex.acquire();
                if (3.0 < bestDist) { bestDist = 3.0; bestLabel = "B"; }
            } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            finally { mutex.release(); }
        }

        @Arbiter
        public void arbiter(L_Result r) {
            r.r1 = bestLabel + ":" + bestDist;
        }
    }

    @JCStressTest
    @Description("Testa a atualização lock-free com Compare-And-Swap (KNNAtomic).")
    @Outcome(id = {"B:3.0", "A:5.0"}, expect = Expect.ACCEPTABLE, desc = "Atualização Atômica OK.")
    @Outcome(id = {"A:3.0", "B:5.0"}, expect = Expect.FORBIDDEN, desc = "FALHA NO CAS!")
    @State
    public static class AtomicState {
        AtomicReference<DistanceRecord> bestData = new AtomicReference<>(new DistanceRecord(10.0, "None"));

        @Actor
        public void actor1() {
            updateIfBetter(5.0, "A");
        }

        @Actor
        public void actor2() {
            updateIfBetter(3.0, "B");
        }

        private void updateIfBetter(double newDist, String newLabel) {
            while (true) {
                DistanceRecord current = bestData.get();
                if (newDist >= current.distance) break;

                DistanceRecord newData = new DistanceRecord(newDist, newLabel);
                if (bestData.compareAndSet(current, newData)) break;
            }
        }

        @Arbiter
        public void arbiter(L_Result r) {
            DistanceRecord result = bestData.get();
            r.r1 = result.label + ":" + result.distance;
        }
    }

    @JCStressTest
    @Description("Testa se ExecutorService.awaitTermination() publica com segurança (sem volatile) a escrita feita dentro da task — a garantia que sustenta o merge do array de resultados em KNNExecutor.")
    @Outcome(id = "42", expect = Expect.ACCEPTABLE, desc = "awaitTermination() publicou a escrita corretamente.")
    @Outcome(id = "0", expect = Expect.FORBIDDEN, desc = "FALHA DE PUBLICAÇÃO! awaitTermination() não estabeleceu happens-before.")
    @Outcome(id = "-1", expect = Expect.ACCEPTABLE_INTERESTING, desc = "Thread interrompida durante o awaitTermination (raro).")
    @State
    public static class ExecutorPublicationState {

        int sideEffectField = 0;

        @Actor
        public void actor(I_Result r) {
            ExecutorService executor = Executors.newSingleThreadExecutor();
            executor.execute(() -> sideEffectField = 42);
            executor.shutdown();
            try {
                executor.awaitTermination(1, TimeUnit.MINUTES);
                r.r1 = sideEffectField;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                r.r1 = -1;
            }
        }
    }

    @JCStressTest
    @Description("Testa se Future.get() publica com segurança (sem volatile) a escrita feita dentro do Callable — a garantia que sustenta o merge em KNNCallableFuture.")
    @Outcome(id = "42", expect = Expect.ACCEPTABLE, desc = "Future.get() publicou a escrita corretamente.")
    @Outcome(id = "0", expect = Expect.FORBIDDEN, desc = "FALHA DE PUBLICAÇÃO! Future.get() não estabeleceu happens-before.")
    @Outcome(id = "-1", expect = Expect.ACCEPTABLE_INTERESTING, desc = "Exceção durante o get() (raro).")
    @State
    public static class CallableFuturePublicationState {

        int sideEffectField = 0;

        @Actor
        public void actor(I_Result r) {
            ExecutorService executor = Executors.newSingleThreadExecutor();
            try {
                Future<Integer> future = executor.submit(() -> {
                    sideEffectField = 42;
                    return null;
                });
                future.get();
                r.r1 = sideEffectField;
            } catch (Exception e) {
                r.r1 = -1;
            } finally {
                executor.shutdown();
            }
        }
    }

    @JCStressTest
    @Description("Testa se ForkJoinTask.join() publica com segurança (sem volatile) a escrita feita dentro da tarefa — a garantia que sustenta o merge lock-free do Top-K em KNNForkJoin.")
    @Outcome(id = "42", expect = Expect.ACCEPTABLE, desc = "join() publicou a escrita corretamente.")
    @Outcome(id = "0", expect = Expect.FORBIDDEN, desc = "FALHA DE PUBLICAÇÃO! join() não estabeleceu happens-before.")
    @State
    public static class ForkJoinPublicationState {

        int sideEffectField = 0;

        @Actor
        public void actor(I_Result r) {
            RecursiveAction task = new RecursiveAction() {
                @Override
                protected void compute() {
                    sideEffectField = 42;
                }
            };
            ForkJoinPool.commonPool().execute(task);
            task.join();
            r.r1 = sideEffectField;
        }
    }

    @JCStressTest
    @Description("Testa se parallelStream()...reduce() publica com segurança (sem volatile) a escrita feita dentro do map() — a garantia que sustenta o merge de Top-Ks em KNNParallelStream.")
    @Outcome(id = "42", expect = Expect.ACCEPTABLE, desc = "reduce() publicou a escrita corretamente.")
    @Outcome(id = "0", expect = Expect.FORBIDDEN, desc = "FALHA DE PUBLICAÇÃO! O pipeline de stream não estabeleceu happens-before.")
    @State
    public static class ParallelStreamPublicationState {

        int sideEffectField = 0;

        @Actor
        public void actor(I_Result r) {
            List<Integer> data = List.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
            
            data.parallelStream()
                    .map(x -> { sideEffectField = 42; return x; })
                    .reduce(0, Integer::sum);
            r.r1 = sideEffectField;
        }
    }

    @JCStressTest
    @Description("Testa se CompletableFuture.join() publica com segurança (sem volatile) a escrita feita dentro do supplyAsync/runAsync — a garantia que sustenta o merge em KNNCompletableFuture.")
    @Outcome(id = "42", expect = Expect.ACCEPTABLE, desc = "join() publicou a escrita corretamente.")
    @Outcome(id = "0", expect = Expect.FORBIDDEN, desc = "FALHA DE PUBLICAÇÃO! CompletableFuture.join() não estabeleceu happens-before.")
    @State
    public static class CompletableFuturePublicationState {

        int sideEffectField = 0;

        @Actor
        public void actor(I_Result r) {
            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> sideEffectField = 42);
            future.join();
            r.r1 = sideEffectField;
        }
    }

    @JCStressTest
    @Description("Testa se StructuredTaskScope.join() publica com segurança (sem volatile) a escrita feita dentro do subtask — a garantia que sustenta o merge lock-free do Top-K em KNNStructuredConcurrency.")
    @Outcome(id = "42", expect = Expect.ACCEPTABLE, desc = "join() publicou a escrita corretamente.")
    @Outcome(id = "0", expect = Expect.FORBIDDEN, desc = "FALHA DE PUBLICAÇÃO! join() não estabeleceu happens-before.")
    @Outcome(id = "-1", expect = Expect.ACCEPTABLE_INTERESTING, desc = "Thread interrompida durante o join (raro).")
    @State
    public static class StructuredConcurrencyPublicationState {

        int sideEffectField = 0;

        @Actor
        public void actor(I_Result r) {
            try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
                scope.fork(() -> {
                    sideEffectField = 42;
                    return null;
                });
                scope.join();
                r.r1 = sideEffectField;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                r.r1 = -1;
            }
        }
    }
}