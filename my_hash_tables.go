package main

// import (
// 	"fmt"
// 	// "strconv"
// )

type Hash_Table struct {
	size int
	data [][][]string
}

func createHashTable(size int) (ht Hash_Table) {
	ht.size = size
	ht.data = make([][][]string, size)
	return ht
}

func (ht *Hash_Table) _hash(key string) (hash int) {
	for i, r := range key {
		hash = (hash + int(r)*i) % ht.size
	}
	return hash
}

func (ht *Hash_Table) set(key string, value string) {
	address := ht._hash(key)
	if len(ht.data[address]) != 0 {
		ht.data[address] = make([][]string, 2)
	}
	pair := []string{key, value}
	ht.data[address] = append(ht.data[address], pair)
}

func (ht *Hash_Table) get(key string) string {
	address := ht._hash(key)
	if len(ht.data[address]) != 0 {
		for _, pair := range ht.data[address] {
			if len(pair) != 0 && pair[0] == key {
				return pair[1]
			}
		}
	}
	return ""
}

func (ht *Hash_Table) keys() (keysList []string) {
	for i := 0; i < ht.size; i++ {
		if ht.data[i] != nil {
			for _, pair := range ht.data[i] {
				keysList = append(keysList, pair[0])
			}
		}
	}
	return keysList
}
