package benchmark;

import org.openjdk.jcstress.annotations.*;
import org.openjdk.jcstress.infra.results.L_Result;

import java.util.concurrent.Semaphore;
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
}