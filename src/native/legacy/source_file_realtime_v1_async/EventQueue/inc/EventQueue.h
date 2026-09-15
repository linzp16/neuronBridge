/*
 * EventQueue.h
 *
 * Binary-heap event queue used by the legacy event-driven simulation. Each
 * worker queue owns a local heap, and a separate synchronization heap stores
 * events that coordinate all queues.
 */
#ifndef EVENTQUEUE_H
#define EVENTQUEUE_H

#include "../source_file_realtime_v1_async/Event/inc/Event.h"

/* Event pointer plus the timestamp used for heap ordering. */
struct Event_in_Queue {
    Event* EventPtr;
    int time;
};

class EventQueue {
public:
    // Per-queue event heaps: Event_Queue_vector[queue][heap_index].
    Event_in_Queue** Event_Queue_vector;

    // Current number of events stored in each queue heap.
    unsigned int* Event_Queue_vector_size;

    // Allocated capacity of each queue heap.
    unsigned int* Event_Queue_vector_capacity;

    // Number of worker/event queues.
    int NumberOfQueue;

    // Heap for synchronization events shared across all worker queues.
    Event_in_Queue* Event_with_syn;

    // Number of synchronization events currently stored.
    int NumberOfEvent_with_syn;

    // Allocated capacity of the synchronization-event heap.
    int Event_with_syn_capacity;

    // Cross-queue event buffers: Buffers[source_queue][target_queue][event_index].
    Event**** Buffers;

    // Current number of buffered events per source/target queue pair.
    int** Buffers_size;

    // Allocated capacity per source/target queue pair.
    int** Buffers_capacity;

    // Creates queue heaps and, when requested, cross-queue buffer storage.
    EventQueue(int numberofQueue, bool initializeStorage = true);

    // Releases queue heaps, synchronization heap, and buffer storage.
    virtual ~EventQueue();

    // Swaps two events inside one worker queue heap.
    void swap_Event(int index, int location1, int location2);

    // Resizes one worker queue heap.
    void Resize_Event_Queue(int index, int Newsize);

    // Inserts an event into one worker queue heap.
    virtual void Insert_a_Event(Event* event, int index);

    // Removes and returns the earliest event from one worker queue heap.
    virtual Event* Remove_a_Event(int index);

    // Returns the current event count for one worker queue heap.
    virtual int Get_Event_Queue_vector_size(int index);

    // Returns the time of the earliest event in one worker queue heap.
    virtual int Get_First_Event_Time(int index);

    // Swaps two events inside the synchronization heap.
    void swap_syn_Event(int location1, int location2);

    // Resizes the synchronization heap.
    void Resize_syn_Event_Queue(int Newsize);

    // Returns the number of events in the synchronization heap.
    virtual int Get_syn_Event_Queue_vector_size();

    // Inserts an event into the synchronization heap.
    virtual void Insert_a_syn_Event(Event* event);

    // Removes and returns the earliest synchronization event.
    virtual Event* Remove_a_syn_Event();

    // Returns the time of the earliest synchronization event.
    virtual int Get_First_syn_Event_Time();

    // Buffers a cross-queue event from source queue index1 to target queue index2.
    virtual void Insert_a_Event_to_Buffer(Event* event, int index1, int index2);

    // Deletes all buffered events targeting the given queue.
    virtual void Reset_Buffer(int index);

    // Moves buffered cross-queue events into the target worker queue heap.
    virtual void Insert_Buffer_to_Event_Queue(int index);

    // Resizes one cross-queue buffer.
    void Resize_Buffer(int index1, int index2);

    // Returns the current number of buffered events for a source/target pair.
    virtual int Get_Buffer_size(int index1, int index2);

    // Returns the allocated capacity for a source/target buffer.
    virtual int Get_Buffer_capacity(int index1, int index2);

    // Increments the buffered-event count for a source/target pair.
    virtual void Increase_Buffer_size(int index1, int index2);

    // Resets the buffered-event count for a source/target pair.
    virtual void ResetSizeBuffer(int index1, int index2);

    // Returns true when all cross-queue buffers are empty.
    virtual bool is_Buffer_Empty();
};

#endif
